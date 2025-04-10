import numpy as np

from .utils import mie_log

MY_CATEGORY = "🐑 MieNodes/🐑 Test"


class BatchDataGenerator(object):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps_per_batch": ("INT", {"default": 5, "min": 1, "max": 1024, "step": 1}),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100000000000.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("BATCH_DATA",)
    RETURN_NAMES = ("batch_data",)
    FUNCTION = "produce"

    CATEGORY = MY_CATEGORY

    def produce(self, steps_per_batch, duration):
        def generator():
            i = 0
            while i < duration:
                # 这里可以添加复杂的生成器逻辑
                yield mie_log(i)
                i += steps_per_batch
        return generator(),


class BatchDataHandler(object):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_data": ("BATCH_DATA",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "handle"

    CATEGORY = MY_CATEGORY

    def handle(self, batch_data):
        logs = []
        for data in batch_data:
            logs.append(mie_log(f"Batch data: {data}"))
        return "\n".join(logs),
