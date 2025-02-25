from .task1 import task_1, task_detail_1
from .task2 import task_2, task_detail_2
from .task3 import task_3, task_detail_3
from .task4 import task_4, task_detail_4
from .task5 import task_5, task_detail_5

class TaskMapper:
    def __init__(self):
        self.task_to_function = {
            1: task_1,
            2: task_2,
            3: task_3,
            4: task_4,
            5: task_5,
        }
        self.task_desc_to_function = {
            1: task_detail_1,
            2: task_detail_2,
            3: task_detail_3,
            4: task_detail_4,
            5: task_detail_5,
        }

    def get_task_function(self, task_id, parameters):
        if task_id in self.task_to_function:
            return self.generate_output(self.task_to_function[task_id](parameters))
        else:
            # Specific error handling?
            return self.generate_output((False, 0, "Invalid task ID", None))

    def get_task_detail(self, task_id):
        if task_id in self.task_desc_to_function:
            return self.task_desc_to_function[task_id]()
        else:
            return None
    
    def generate_output(self, output_tuple):
        return {
            "success": output_tuple[0],
            "score": output_tuple[1],
            "message": output_tuple[2],
            "image": output_tuple[3], ## TODO: add image to cloud, get url
        }

taskmapper = TaskMapper()