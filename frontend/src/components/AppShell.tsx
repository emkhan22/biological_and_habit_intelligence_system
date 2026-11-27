import React, { useState } from "react";
import {
  Activity,
  BookOpen,
  Flame,
  Lock,
  Moon,
  Scale,
  ShieldAlert,
  Sparkles,
  Stethoscope,
  Unlock,
  X,
} from "lucide-react";
import type { NavTabId } from "../types";

interface Props {
  activeTab: NavTabId;
  onSelectTab: (tab: NavTabId) => void;
  privateUnlocked: boolean;
  onUnlockPrivate: (email: string, pin: string) => Promise<void>;
  onLockPrivate: () => void;
  syncTimestampIst?: string;
  dataSourceLabel?: string;
  children: React.ReactNode;
}

const CLEAN_NAV_ITEMS: Array<{ id: NavTabId; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { id: "today", label: "Today", icon: Sparkles },
  { id: "body_fuel", label: "Body & Fuel", icon: Scale },
  { id: "sleep_energy", label: "Sleep & Energy", icon: Moon },
  { id: "therapies_reports", label: "Therapies & Reports", icon: Stethoscope },
  { id: "ask_research", label: "Ask & Research", icon: BookOpen },
];

export const AppShell: React.FC<Props> = ({
  activeTab,
  onSelectTab,
  privateUnlocked,
  onUnlockPrivate,
  onLockPrivate,
  syncTimestampIst,
  dataSourceLabel,
  children,
}) => {
  const [modalOpen, setModalOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [pin, setPin] = useState("");
  const [unlockError, setUnlockError] = useState<string | null>(null);
  const [unlocking, setUnlocking] = useState(false);

  const navItems = privateUnlocked
    ? [...CLEAN_NAV_ITEMS, { id: "cravings" as NavTabId, label: "Cravings", icon: Flame }]
    : CLEAN_NAV_ITEMS;

  async function handleUnlockSubmit(e: React.FormEvent) {
    e.preventDefault();
    setUnlockError(null);
    setUnlocking(true);
    try {
      await onUnlockPrivate(email.trim(), pin.trim());
      setPin("");
      setModalOpen(false);
      onSelectTab("cravings");
    } catch (err: any) {
      setUnlockError(err?.message || "Unlock failed.");
    } finally {
      setUnlocking(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#0B0D12] text-[#F1F5F9] flex flex-col md:flex-row">
      {/* Desktop Left Sidebar Rail */}
      <aside className="no-print hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0 border-r border-[#283042] bg-[#141821] px-4 py-5">
        <div className="flex items-center gap-2.5 px-2 pb-5 border-b border-[#283042]">
          <Activity className="w-5 h-5 text-[#C06C9F]" />
          <div>
            <p className="text-sm font-semibold tracking-tight text-slate-100">Health Intelligence</p>
            <p className="text-[11px] text-slate-400">V3.0 · Store V2 (IST)</p>
          </div>
        </div>

        <nav className="mt-4 flex-1 space-y-1" aria-label="Primary Navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectTab(item.id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
                  active
                    ? "bg-[#1B202C] text-slate-100 border border-[#283042]"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#1B202C]/50 border border-transparent"
                }`}
              >
                <Icon className={`w-4 h-4 ${active ? "text-[#C06C9F]" : "text-slate-500"}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="pt-4 border-t border-[#283042] space-y-2.5">
          {privateUnlocked ? (
            <button
              type="button"
              onClick={onLockPrivate}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium bg-[#FB7185]/10 text-[#FB7185] border border-[#FB7185]/30 hover:bg-[#FB7185]/20 cursor-pointer"
            >
              <Unlock className="w-3.5 h-3.5" />
              Lock Private Mode
            </button>
          ) : (
            <button
              type="button"
              onClick={() => setModalOpen(true)}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium bg-[#1B202C] text-slate-300 border border-[#283042] hover:text-white cursor-pointer"
            >
              <Lock className="w-3.5 h-3.5 text-slate-400" />
              Private Mode
            </button>
          )}
          <div className="px-2 text-[11px] text-slate-500 space-y-0.5">
            <p className="truncate">{dataSourceLabel || "Health Intelligence Store V2"}</p>
            {syncTimestampIst ? <p className="font-mono text-[10px]">Sync: {syncTimestampIst}</p> : null}
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 md:pl-64 flex flex-col min-h-screen">
        {/* Mobile Top Bar */}
        <header className="no-print md:hidden sticky top-0 z-20 flex items-center justify-between px-4 py-3 bg-[#141821] border-b border-[#283042]">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#C06C9F]" />
            <span className="text-sm font-semibold">Health Intelligence V3</span>
          </div>
          {privateUnlocked ? (
            <button
              type="button"
              onClick={onLockPrivate}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs bg-[#FB7185]/10 text-[#FB7185] border border-[#FB7185]/30"
            >
              <Unlock className="w-3 h-3" /> Lock
            </button>
          ) : (
            <button
              type="button"
              onClick={() => setModalOpen(true)}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs bg-[#1B202C] text-slate-300 border border-[#283042]"
            >
              <Lock className="w-3 h-3" /> Private
            </button>
          )}
        </header>

        <main className="flex-1 px-4 py-5 md:px-7 md:py-6 pb-24 md:pb-10 max-w-7xl w-full mx-auto">
          {children}
        </main>

        {/* Mobile Bottom Navigation Bar */}
        <nav
          className="no-print md:hidden fixed bottom-0 inset-x-0 z-20 bg-[#141821] border-t border-[#283042] flex items-center justify-around px-1 py-1.5"
          aria-label="Mobile Navigation"
        >
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectTab(item.id)}
                className={`flex flex-col items-center gap-0.5 px-2 py-1 rounded text-[10px] font-medium ${
                  active ? "text-[#C06C9F]" : "text-slate-400"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="truncate max-w-[64px]">{item.label.split(" ")[0]}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Private Mode Secondary PIN Unlock Modal */}
      {modalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
          role="dialog"
          aria-modal="true"
          aria-label="Unlock Private Mode"
        >
          <div className="obsidian-card w-full max-w-sm p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-[#FB7185]" />
                <h2 className="text-sm font-semibold">Unlock Private Mode</h2>
              </div>
              <button
                type="button"
                onClick={() => setModalOpen(false)}
                className="text-slate-400 hover:text-white cursor-pointer"
                aria-label="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Strictly isolated behind Cloudflare Access email verification + secondary PIN. Reads only{" "}
              <code className="font-mono text-slate-300">Smoking_Private_Store</code>; zero data is stored in{" "}
              <code className="font-mono text-slate-300">Health Intelligence Store V2</code>.
            </p>
            <form onSubmit={handleUnlockSubmit} className="space-y-3">
              <div>
                <label htmlFor="private-email" className="block text-xs text-slate-300 mb-1">
                  Allowlisted Email
                </label>
                <input
                  id="private-email"
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full rounded-lg bg-[#0B0D12] border border-[#283042] px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-[#C06C9F]"
                />
              </div>
              <div>
                <label htmlFor="private-pin" className="block text-xs text-slate-300 mb-1">
                  Secondary PIN
                </label>
                <input
                  id="private-pin"
                  type="password"
                  required
                  value={pin}
                  onChange={(e) => setPin(e.target.value)}
                  placeholder="Enter PIN"
                  className="w-full rounded-lg bg-[#0B0D12] border border-[#283042] px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-[#C06C9F]"
                />
              </div>
              {unlockError ? (
                <p className="text-xs text-[#FB7185]">{unlockError}</p>
              ) : null}
              <button
                type="submit"
                disabled={unlocking}
                className="w-full py-2 rounded-lg bg-[#C06C9F] text-[#0B0D12] font-semibold text-sm hover:opacity-90 disabled:opacity-50 cursor-pointer"
              >
                {unlocking ? "Verifying..." : "Unlock Private View"}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
