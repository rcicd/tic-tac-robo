import sys
import os
import time
import threading
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, Session_pb2
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager

TIMEOUT_DURATION = 20

class DeviceConnection:

    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection(ip = "192.168.1.10",
                            username = "admin",
                            password = "admin"): 
        return DeviceConnection(ip, port=DeviceConnection.TCP_PORT, credentials=(username, password))

    @staticmethod
    def createUdpConnection(args): 
        return DeviceConnection(args.ip, port=DeviceConnection.UDP_PORT, credentials=(args.username, args.password))

    def __init__(self, ipAddress, port=TCP_PORT, credentials = ("","")):

        self.ipAddress = ipAddress
        self.port = port
        self.credentials = credentials

        self.sessionManager = None

        # Setup API
        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    # Called when entering 'with' statement
    def __enter__(self):
        
        self.transport.connect(self.ipAddress, self.port)

        if (self.credentials[0] != ""):
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000   # (milliseconds)
            session_info.connection_inactivity_timeout = 2000 # (milliseconds)

            self.sessionManager = SessionManager(self.router)
            print("Logging as", self.credentials[0], "on device", self.ipAddress)
            self.sessionManager.CreateSession(session_info)

        return self.router

    # Called when exiting 'with' statement
    def __exit__(self, exc_type, exc_value, traceback):
    
        if self.sessionManager != None:

            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000 
            
            self.sessionManager.CloseSession(router_options)

        self.transport.disconnect()



def check_for_end_or_abort(e):
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check


def move_to_home(base):
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def cartesian_action_movement(base, 
                              base_cyclic, 
                              x = 0, y = 0, z = 0, 
                              tx = 0, ty = 0, tz = 0):
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x + x 
    cartesian_pose.y = feedback.base.tool_pose_y + y 
    cartesian_pose.z = feedback.base.tool_pose_z + z 
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x+tx
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y+ty
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z+tz

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def set_gripper(base, percent):
    if not (0.0 <= percent <= 100.0):
        raise ValueError("percent must be between 0 and 100")

    # Convert percent to [0.0, 1.0] range for the gripper command
    position = percent / 100.0

    gripper_command = Base_pb2.GripperCommand()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger = gripper_command.gripper.finger.add()
    finger.finger_identifier = 1
    finger.value = position

    print(f"Setting gripper to {percent:.1f}% (position={position:.2f})")
    base.SendGripperCommand(gripper_command)

    gripper_request = Base_pb2.GripperRequest()
    gripper_request.mode = Base_pb2.GRIPPER_POSITION

    timeout = 5
    t0 = time.time()
    while True:
        gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
        if len(gripper_measure.finger):
            current = gripper_measure.finger[0].value
            print(f"Current gripper position: {current:.2f}")
            # Allow some epsilon for floating-point precision
            if abs(current - position) < 0.02:
                break
        else:
            print("No finger measurement received.")
            break
        if time.time() - t0 > timeout:
            print("Timeout waiting for gripper to reach position.")
            break
        time.sleep(0.1)
    return 1

if __name__ == "__main__":
    with DeviceConnection.createTcpConnection() as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        move_to_home(base)

