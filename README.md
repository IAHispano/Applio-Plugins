# Applio Plugins

Welcome to **Applio Plugins**, a repository specifically designed for Applio plugins.

If you're not familiar with Applio, check it out on our incredible webpage, [applio.org](https://applio.org), or visit our [GitHub repository](https://github.com/IAHispano/Applio).

## If you are going to create a new plugin

The heart of the plugin lies in the `def applio_plugin()` function, acting as the interface for the Gradio tab. This function will be brought into the plugins tab later on. It's crucial to maintain the original names of both the function and the `plugin.py` file, as they are integral to the import process. Additionally, there's a requirements file that cannot be relocated or renamed but can be removed if not needed.
