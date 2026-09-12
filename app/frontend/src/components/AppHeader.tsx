import {ArrowRight} from 'lucide-react';

import type {HealthResponse} from '../types';
import StatusPill from './StatusPill';


export default function AppHeader({health}: {health: HealthResponse | null}) {
  return (
    <header>
      <div className="brand">
        <div className="logo">V</div>
        <div>
          <h1>VeriSQL Studio</h1>
          <p>Execution-Verified Text-to-SQL Workspace</p>
        </div>
      </div>
      <div className="status">
        <StatusPill good={Boolean(health?.api_ready)}>
          API {health?.api_ready ? 'ready' : 'offline'}
        </StatusPill>
        <StatusPill good={Boolean(health?.model_ready)}>
          Model {health?.model_ready ? 'ready' : 'standby'}
        </StatusPill>
        <span className="about">About <ArrowRight size={14} /></span>
      </div>
    </header>
  );
}
