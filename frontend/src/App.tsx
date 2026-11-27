import React, { useEffect, useState } from "react";
import { AppShell } from "./components/AppShell";
import { fetchCleanHealthData, unlockPrivateSmokingApi } from "./services/api";
import type { HealthDataPayload, NavTabId, PrivateSmokingPayload } from "./types";
import { AskResearchView } from "./views/AskResearchView";
import { BodyFuelView } from "./views/BodyFuelView";
import { CravingsPrivateView } from "./views/CravingsPrivateView";
import { SleepEnergyView } from "./views/SleepEnergyView";
import { TherapiesReportsView } from "./views/TherapiesReportsView";
import { TodayView } from "./views/TodayView";

interface Props {
  initialData?: HealthDataPayload | null;
}

export const App: React.FC<Props> = ({ initialData = null }) => {
  const [activeTab, setActiveTab] = useState<NavTabId>("today");
  const [data, setData] = useState<HealthDataPayload | null>(initialData);
  const [error, setError] = useState<string | null>(null);
  const [privateUnlocked, setPrivateUnlocked] = useState(false);
  const [privateData, setPrivateData] = useState<PrivateSmokingPayload | null>(null);

  useEffect(() => {
    if (initialData) return;
    let cancelled = false;
    fetchCleanHealthData()
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message || "Failed to load Health Intelligence Store V2.");
      });
    return () => {
      cancelled = true;
    };
  }, [initialData]);

  async function handleUnlockPrivate(email: string, pin: string) {
    const priv = await unlockPrivateSmokingApi(email, pin);
    setPrivateData(priv);
    setPrivateUnlocked(true);
  }

  function handleLockPrivate() {
    setPrivateUnlocked(false);
    setPrivateData(null);
    if (activeTab === "cravings") {
      setActiveTab("today");
    }
  }

  return (
    <AppShell
      activeTab={activeTab}
      onSelectTab={setActiveTab}
      privateUnlocked={privateUnlocked}
      onUnlockPrivate={handleUnlockPrivate}
      onLockPrivate={handleLockPrivate}
      syncTimestampIst={data?.packet?.meta?.sync_timestamp_ist}
      dataSourceLabel={data?.source}
    >
      {error ? (
        <div className="obsidian-card p-6 text-sm text-[#FB7185]">{error}</div>
      ) : !data ? (
        <div className="obsidian-card p-6 text-sm text-slate-400">
          Loading validated telemetry from Health Intelligence Store V2...
        </div>
      ) : (
        <>
          {activeTab === "today" && <TodayView data={data} />}
          {activeTab === "body_fuel" && <BodyFuelView data={data} />}
          {activeTab === "sleep_energy" && <SleepEnergyView data={data} />}
          {activeTab === "therapies_reports" && <TherapiesReportsView data={data} />}
          {activeTab === "ask_research" && <AskResearchView data={data} />}
          {activeTab === "cravings" && privateUnlocked && (
            <CravingsPrivateView privateData={privateData} />
          )}
        </>
      )}
    </AppShell>
  );
};

export default App;
