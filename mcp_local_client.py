from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner

import json
import asyncio
import nest_asyncio
import ollama



nest_asyncio.apply()

class MCP_ChatBot:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        # Tool list
        self.available_tools = []
        # Prompts list for quick display 
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions = {}
        # Rich console
        self.console = Console()

    async def connect_to_server(self, server_name, server_config):
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            
            try:
                # List available tools
                response = await session.list_tools()
                for tool in response.tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": tool.inputSchema.get("type"),
                                "required": tool.inputSchema.get("required"),
                             },
                            "properties": tool.inputSchema.get("properties"),
                        }
                    })

                # List available prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        })
                # List available resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
            
            except Exception as e:
                self.console.print(f"Error {e}")
                
        except Exception as e:
            self.console.print(f"Error connecting to {server_name}: {e}")

    async def connect_to_servers(self):
        try:
            with open("mcp_local_server_config.json", "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            self.console.print(f"Error loading server config: {e}")
            raise
    
    async def process_query(self, query):
        messages = [{'role':'user', 'content':query}]
        #self.console.print(f"Tools Available\n {json.dumps(self.available_tools, indent=4)}")

        while True:
            response = ollama.chat(
                model = 'qwen3:8b-q4_K_M',
                tools = self.available_tools,
                messages = messages,
            )
            
            assistant_content = []
            has_tool_use = False

            self.console.print(response.message.content)
            assistant_content.append(response.message.content)

            for tool in response.message.tool_calls or []:
                has_tool_use = True
                messages.append({
                    "role": "assistant", 
                    "content": "".join(assistant_content)
                })

                # Get session and call tool
                self.console.print(f"\n[bold green]Tool Call:[/bold green] {tool.function}\n")
                session = self.sessions.get(tool.function.name)
                if not session:
                    self.console.print(f"Tool '{tool.function.name}' not found.")
                    break
                    
                result = await session.call_tool(tool.function.name, arguments=tool.function.arguments)

                tool_output = []
                for content in result.content:
                    tool_output.append(content.text)
                self.console.print(f"\n[bold magenta]Tool Output:[/bold magenta] {tool_output}\n")
         
                messages.append({
                    "role": "tool", 
                    "content": "".join(tool_output)
                })
            
            # Exit loop if no tool was used
            if not has_tool_use:
                break

    async def get_resource(self, resource_uri):
        session = self.sessions.get(resource_uri)
        
        # Fallback for papers URIs - try any papers resource session
        if not session and resource_uri.startswith("papers://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break
            
        if not session:
            self.console.print(f"Resource '{resource_uri}' not found.")
            return
        
        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                self.console.print(f"\nResource: {resource_uri}")
                self.console.print("Content:")
                self.console.print(result.contents[0].text)
            else:
                self.console.print("No content available.")
        except Exception as e:
            self.console.print(f"Error: {e}")
    
    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            self.console.print("No prompts available.")
            return
        
        self.console.print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            self.console.print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                self.console.print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    self.console.print(f"    - {arg_name}")
    
    async def execute_prompt(self, prompt_name, args):
        """Execute a prompt with the given arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            self.console.print(f"Prompt '{prompt_name}' not found.")
            return
        
        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content
                
                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(item.text if hasattr(item, 'text') else str(item) 
                                  for item in prompt_content)
                
                self.console.print(f"\nExecuting prompt '{prompt_name}'...")
                await self.process_query(text)
        except Exception as e:
            self.console.print(f"Error: {e}")
    
    async def chat_loop(self):
        self.console.print(Panel.fit("Local MCP Client Started!", padding=(1,4)))
        self.console.print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
        
                if query.lower() == 'quit':
                    break
                
                await self.process_query(query)
                    
            except Exception as e:
                self.console.print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
