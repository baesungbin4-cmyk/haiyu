import Dashboard from "./pages/Dashboard";
import { useLiveDashboardData } from "./services/liveData";
import { useRealtimeStatus } from "./services/realtime";
import "./styles/dashboard.css";

export default function App() {
  const realtimeStatus = useRealtimeStatus();
  const dashboardData = useLiveDashboardData(realtimeStatus.lastMessage);

  return (
    <div className="app-shell">
      <Dashboard data={dashboardData} realtimeStatus={realtimeStatus} />
    </div>
  );
}
