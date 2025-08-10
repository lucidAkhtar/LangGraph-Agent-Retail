1. Separation of Concerns

- schemas.py -> request/response validation
- services/ -> business logic (can be tested independentally)
- routes.py -> HTTP layer only
- agents/ -> your core ML logic


2. Non-blocking execution

- run_in_executor ensures your sync heavy code does not block the event loop.
- Makes the API scale better under concurrent load.

3. Testability

- You can unit test your business logic without touching FastAPI

4. Extensibility

- Adding new routes or swapping out the agent implementation doesn't touch the API layer.
 