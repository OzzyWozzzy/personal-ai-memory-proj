# Package marker. Router modules dropped into this package are auto-discovered
# by app.main.create_app(): any submodule exposing a module-level `router`
# (fastapi.APIRouter) is included automatically, behind the X-API-Key
# dependency. No edits to main.py are needed to add an endpoint.
