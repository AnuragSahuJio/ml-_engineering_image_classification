import time
import asyncio
from functools import wraps

def monitor_prediction_time(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)  # Await async functions
        end_time = time.time()
        print(f"Prediction time: {end_time - start_time:.4f} seconds")
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Call synchronous functions
        end_time = time.time()
        print(f"Prediction time: {end_time - start_time:.4f} seconds")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
