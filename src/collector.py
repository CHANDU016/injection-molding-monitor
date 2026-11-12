"""Where hourly data comes from.

DummyCollector -> simulated data shaped like your CSV (for building/testing).
RealCollector  -> YOU fill in read_cycles() for your machine (OPC UA / Modbus / MQTT / DB).
Switch with `mode:` in config.yaml. Nothing else in the project changes.
"""
import numpy as np
import pandas as pd
from .config import load, path


class DummyCollector:
    source = "dummy"

    def __init__(self, cfg=None):
        self.cfg = cfg or load()
        self.params = self.cfg["parameters"]
        seed = pd.read_csv(path(self.cfg["seed_csv"]))
        X = seed[self.params]
        self.mean = X.mean().values
        self.cov = X.cov().values
        rng = X.max() - X.min()
        self.lo = (X.min() - 0.1 * rng).values
        self.hi = (X.max() + 0.1 * rng).values
        self.p_defect = float((seed["Quality_Result"] == "Defective").mean())
        self.rng = np.random.default_rng()

    def read_cycles(self, ts_end, fault=None):
        n = int(self.cfg["cycles_per_hour"])
        X = self.rng.multivariate_normal(self.mean, self.cov, size=n)
        df = pd.DataFrame(X, columns=self.params)

        # optional fault injection to test alerts / anomaly detection
        if fault == "oil_overheat":      # hydraulic oil getting hot
            df["Hydraulic_Oil_Temperature_C"] += np.linspace(5, 18, n)
        elif fault == "pressure_drop":   # pump / valve problem
            df["Injection_Pressure_bar"] -= 30
            df["Holding_Pressure_bar"] -= 15
        elif fault == "heater_fail":     # heater band failing
            df["Melt_Temperature_C"] -= np.linspace(2, 20, n)

        df = df.clip(self.lo, self.hi, axis=1) if fault is None else df
        ts_end = pd.Timestamp(ts_end)
        step = pd.Timedelta(seconds=3600 / n)
        df.insert(0, "ts", [ts_end - pd.Timedelta(hours=1) + step * (i + 1) for i in range(n)])
        df["quality_result"] = np.where(self.rng.random(n) < self.p_defect, "Defective", "Good")
        df["source"] = self.source
        df[self.params] = df[self.params].round(3)
        return df


class RealCollector:
    source = "real"

    def __init__(self, cfg=None):
        self.cfg = cfg or load()

    def read_cycles(self, ts_end, fault=None):
        """Return a DataFrame with columns: ts, <all parameters in config>,
        quality_result (None if not inspected yet), source='real'.

        Example with OPC UA (pip install opcua):
            from opcua import Client
            c = Client("opc.tcp://192.168.1.10:4840"); c.connect()
            node = c.get_node("ns=2;s=Machine.MeltTemp"); value = node.get_value()

        Example with Modbus TCP (pip install pymodbus):
            from pymodbus.client import ModbusTcpClient
            m = ModbusTcpClient("192.168.1.10"); regs = m.read_holding_registers(0, 10)
        """
        raise NotImplementedError("Implement RealCollector.read_cycles() for your machine.")


def get_collector(cfg=None):
    cfg = cfg or load()
    return DummyCollector(cfg) if cfg["mode"] == "dummy" else RealCollector(cfg)
