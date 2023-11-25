import multiprocessing
import time
from typing import Callable, Any, Generator, Tuple


class TaskManager:
    def __init__(
        self,
        num_consumers: int,
        produce_new_task: Generator[Tuple[int, str], Any, None],
        consumer_func: Callable[[Any], Any],
        process_finished_task: Callable,
    ):
        self.num_consumers = num_consumers
        self.produce_new_task = produce_new_task
        self.consumer_func = consumer_func
        self.process_finished_task = process_finished_task
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

    def run(self) -> None:
        # Start consumer processes
        consumers = [
            multiprocessing.Process(
                target=self.consumer_wrapper, args=(self.input_queue, self.output_queue)
            )
            for _ in range(self.num_consumers)
        ]
        for c in consumers:
            c.start()

        # Start the producer process
        producer_process = multiprocessing.Process(
            target=self.producer_wrapper, args=(self.input_queue, self.output_queue)
        )
        producer_process.start()

        # Wait for the producer to finish
        producer_process.join()

        # Wait for all consumers to finish
        for c in consumers:
            c.join()

    def producer_wrapper(
        self, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue
    ) -> None:
        tasks_completed = 0
        work_production_finished = False
        finished_tasks_processed = False

        while not finished_tasks_processed:
            if not work_production_finished:
                try:
                    task = next(self.produce_new_task)
                    input_queue.put(task)
                except StopIteration:
                    work_production_finished = True
                    # Signal consumers that no new work will arrive
                    for _ in range(self.num_consumers):
                        input_queue.put(None)

            while not output_queue.empty():
                result = output_queue.get()
                if result is None:
                    tasks_completed += 1
                    if tasks_completed == self.num_consumers:
                        finished_tasks_processed = True
                        self.process_finished_task(None, all_tasks_finished=True)
                        break
                else:
                    self.process_finished_task(result, all_tasks_finished=False)

    def consumer_wrapper(
        self, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue
    ) -> None:
        while True:
            task = input_queue.get()
            if task is None:
                output_queue.put(None)
                break
            result = self.consumer_func(task)
            output_queue.put(result)
