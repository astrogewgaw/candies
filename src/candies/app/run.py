from cyclopts import App

app = App()

app["--help"].group = "Admin"
app["--version"].group = "Admin"

app.command("candies.app.commands.make:make")
app.command("candies.app.commands.wrap:wrap")
app.command("candies.app.commands.plot:plot")
app.command("candies.app.commands.label:label")
app.command("candies.app.commands.store:store")
app.command("candies.app.commands.list:list_", name="list")

__all__ = ["app"]

if __name__ == "__main__":
    app()
