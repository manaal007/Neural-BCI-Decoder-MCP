import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    print("=== Step 1: Initializing MCP Client & Connecting over Stdio ===")

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("[SUCCESS] Session established with MCP Server!")

            tools_response = await session.list_tools()
            tools = tools_response.tools
            print(f"\n=== Step 2: Tool Discovery ===")
            print(f"Discovered {len(tools)} Tool(s): {[t.name for t in tools]}")

            sample_signal = [0.12, -0.45, 0.89, 0.34, -0.12, 0.67, 0.05, -0.22, 0.41, -0.15]
            print(f"\n=== Step 3: Invoking MCP Tool '{tools[0].name}' ===")

            result = await session.call_tool(
                name="calculate_eeg_metrics",
                arguments={"signal_values": sample_signal, "sampling_rate": 160}
            )

            print("\n==================================================")
            print("           FINAL MCP SERVER RESULT OUTPUT         ")
            print("==================================================")
            for content in result.content:
                print(content.text)


if __name__ == "__main__":
    asyncio.run(main())