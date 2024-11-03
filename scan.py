import pandas as pd
import numpy as np
import datetime as dt

from arbiscan import ARBISCAN


BEBOP_ADDRESS = "0x51c72848c68a965f66fa7a88855f9f7784502a7f"
JAM_ADDRESS = "0xbEbEbEb035351f58602E0C1C8B59ECBfF5d5f47b"


class ARBISCAN_BEBOP(ARBISCAN):
    def __init__(self):
        super().__init__()
        self.start = dt.datetime(2024, 6, 7, 12, tzinfo=dt.timezone.utc)

    def _check_times(self, time: dt.datetime) -> dt.datetime | None:
        return time if time < dt.datetime.now(dt.timezone.utc) else None

    def get_phase_start_time(self, phase: int) -> dt.datetime | None:
        return self._check_times(self.start + dt.timedelta(weeks=(3 * (phase - 1))))

    def get_phase_end_time(self, phase: int) -> dt.datetime | None:
        return self._check_times(self.start + dt.timedelta(weeks=(3 * phase)))

    def get_phase_start_block(self, phase: int) -> int | None:
        time = self.get_phase_start_time(phase)
        if time is None: return None
        return self.get_block_number_by_time(int(time.timestamp()))

    def get_phase_end_block(self, phase: int) -> int | None:
        time = self.get_phase_end_time(phase)
        if time is None: return None
        return self.get_block_number_by_time(int(time.timestamp()))

    def get_all_bebop_trades(self, address: str, phase: int) -> list:
        resp = self.query_arbiscan(address=address, startblock=self.get_phase_start_block(phase), endblock=self.get_phase_end_block(phase))
        return [result for result in resp if result["to"] == BEBOP_ADDRESS or result["from"] == BEBOP_ADDRESS or result["to"] == JAM_ADDRESS or result["from"] == JAM_ADDRESS]

    def _tidy_df(self, df: pd.DataFrame, extra: bool, time_cutoff: int=30) -> pd.DataFrame:
        df["value"] = df["value"].astype(float)
        df["timeStamp"] = df["timeStamp"].astype(int)
        df["timeStamp_dt"] = pd.to_datetime(df["timeStamp"], unit="s")
        df["tokenDecimal"] = df["tokenDecimal"].astype(int)
        df["price"] = df["value"] / (10 ** df["tokenDecimal"])
        return df if not extra else self._weth_stable_data(df, time_cutoff)

    def _weth_stable_data(self, df: pd.DataFrame, time_cutoff: int) -> pd.DataFrame:
        USDC_df = df[(df["tokenSymbol"] == "USDC") | (df["tokenSymbol"] == "USDT")].copy()
        WETH_df = df[df["tokenSymbol"] == "WETH"].copy()
        data_columns = pd.DataFrame(index=df.index)

        for frame, label in [(USDC_df, "USDC"), (WETH_df, "WETH")]:
            calculated_value = ((frame.shift(-1)["value"] - frame["value"]) / frame["value"])
            frame["timeDelta"] = frame["timeStamp"].diff(-1)
            condition = (frame["timeDelta"] < time_cutoff) & (frame.shift(-1)["timeDelta"] < time_cutoff)
            frame["data"] = np.where(condition, calculated_value, np.nan)
            data_columns[label + "_timeDelta"] = frame["timeDelta"]
            data_columns[label + "_data"] = frame["data"]

        df = df.join(data_columns)
        df["WETH_data"] = df["WETH_data"].replace(0, np.nan)
        df["USDC_data"] = df["USDC_data"].replace(0, np.nan)
        df["timeDelta"] = df.apply(lambda row: df.loc[row.name, "USDC_timeDelta"] if row["tokenSymbol"] == "USDC"
                                    else (df.loc[row.name, "WETH_timeDelta"] if row["tokenSymbol"] == "WETH" else np.nan), axis=1)
        df["data"] = df.apply(lambda row: df.loc[row.name, "USDC_data"] if row["tokenSymbol"] == "USDC"
                            else (df.loc[row.name, "WETH_data"] if row["tokenSymbol"] == "WETH" else np.nan), axis=1)

        return df

    def get_all_bebop_trades_df(self, address: str, phase: int, extra: bool=True, time_cutoff: int=30) -> pd.DataFrame:
        trades = self.get_all_bebop_trades(address, phase)
        df = pd.DataFrame(trades)
        if len(df) == 0: return df
        return self._tidy_df(df, extra, time_cutoff)

    def all_trades(self, addresses: dict[str, str], phase: int, extra: bool=True, time_cutoff: int=30) -> pd.DataFrame:
        df = pd.DataFrame()
        for label, address in addresses.items():
            trades = self.get_all_bebop_trades_df(address, phase, extra, time_cutoff)
            trades["label"] = label
            df = pd.concat([df, trades])
        df.reset_index(drop=True)
        return df
