from Viewer.controller import SimpleViewer, Controller
from Viewer.mesh_viewer import MeshViewer
from character_controller import *

class InteractiveUpdate():
    def __init__(self, viewer, controller, character_controller):
        self.viewer = viewer
        self.controller = controller
        self.character_controller = character_controller
        
    def update(self, task):
        desired_pos_list, desired_rot_list, desired_vel_list, desired_avel_list, current_gait = \
            self.controller.get_desired_state()
        character_state = self.character_controller.update_state(
                desired_pos_list, desired_rot_list, 
                desired_vel_list, desired_avel_list
                )
        self.character_controller.sync_controller_and_character(character_state[1][0])
        for i in range(len(character_state[0])):
            name, pos, rot = character_state[0][i], character_state[1][i], character_state[2][i]
            self.viewer.set_joint_position_orientation(name, pos, rot)
        return task.cont    

def main():
    # viewer = MeshViewer()
    # viewer.run()
    viewer = SimpleViewer()
    controller = Controller(viewer)
    character_controller = CharacterController(viewer, controller)
    task = InteractiveUpdate(viewer, controller, character_controller)
    viewer.addTask(task.update)
    viewer.run()
    pass

if __name__ == '__main__':
    main()