from mcp.server.fastmcp import FastMCP
import numpy as np

# Initialize FastMCP Server
mcp = FastMCP("NeuralSignalProcessor")


@mcp.tool()
def calculate_eeg_metrics(signal_values: list[float], sampling_rate: int = 160) -> dict:
    """Calculates mean power, variance, and peak amplitude for a sample EEG signal epoch."""
    data = np.array(signal_values)
    mean_power = float(np.mean(np.square(data)))
    variance = float(np.var(data))
    peak_amplitude = float(np.max(np.abs(data)))

    return {
        "status": "success",
        "sample_count": len(signal_values),
        "mean_power_uV2": round(mean_power, 4),
        "variance": round(variance, 4),
        "peak_amplitude_uV": round(peak_amplitude, 4)
    }


if __name__ == "__main__":
    mcp.run()