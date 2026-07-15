import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import StatusPage from './pages/StatusPage.jsx'
import AgentPage from './pages/ChatPage.jsx'
import SkillsPage from './pages/SkillsPage.jsx'
import ConfigPage from './pages/ConfigPage.jsx'
import CronsPage from './pages/CronsPage.jsx'
import AlertsPage from './pages/AlertsPage.jsx'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/status" replace />} />
        <Route path="/status" element={<StatusPage />} />
        <Route path="/agent" element={<AgentPage />} />
        <Route path="/agent/:conversationId" element={<AgentPage />} />
        <Route path="/chat/*" element={<Navigate to="/agent" replace />} />
        <Route path="/skills" element={<SkillsPage />} />
        <Route path="/config" element={<ConfigPage />} />
        <Route path="/crons" element={<CronsPage />} />
        <Route path="/alerts" element={<AlertsPage />} />
      </Routes>
    </Layout>
  )
}
