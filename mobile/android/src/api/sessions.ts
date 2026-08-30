import { api, ApiError } from './client';

export type SessionType = 'chat' | 'workspace' | 'container';

export interface SessionLoadProblem {
  code: string;
  title: string;
  message: string;
  resolutionSteps: string[];
  technicalDetail: string;
}

export function describeSessionLoadError(error: unknown): SessionLoadProblem {
  let detail: unknown = error;
  if (error instanceof ApiError && error.body && typeof error.body === 'object') {
    detail = (error.body as Record<string, unknown>).detail ?? error.body;
  }
  if (detail && typeof detail === 'object') {
    const value = detail as Record<string, unknown>;
    return {
      code: String(value.code || 'session_load_failed'),
      title: String(value.title || 'Session could not be loaded'),
      message: String(value.message || (error instanceof Error ? error.message : 'The session could not be loaded.')),
      resolutionSteps: Array.isArray(value.resolution_steps) ? value.resolution_steps.map(String) : [],
      technicalDetail: String(value.technical_detail || ''),
    };
  }
  return {
    code: 'session_load_failed',
    title: 'Session could not be loaded',
    message: error instanceof Error ? error.message : String(error || 'Unknown session load error'),
    resolutionSteps: [],
    technicalDetail: '',
  };
}

export function formatSessionLoadProblem(problem: SessionLoadProblem): string {
  const lines = [problem.message];
  if (problem.resolutionSteps.length) {
    lines.push('', 'Resolution:', ...problem.resolutionSteps.map((step, index) => `${index + 1}. ${step}`));
  }
  if (problem.technicalDetail) lines.push('', `Technical detail: ${problem.technicalDetail}`);
  return lines.join('\n');
}

export interface ContainerMount {
  host_path: string;
  container_path: string;
  mode: 'ro' | 'rw';
}

export interface ContainerDevice {
  host_path: string;
  container_path: string;
  permissions: 'r' | 'rw' | 'rwm';
}

export interface HostGpuDevice {
  id: string;
  index?: string;
  name: string;
  uuid?: string;
}

export interface HostDeviceCandidate extends ContainerDevice {
  kind?: string;
  name?: string;
}

export interface ContainerHardwareCapabilities {
  docker_available: boolean;
  gpu: {
    supported: boolean;
    runtime_detected: boolean;
    docker_runtimes?: string[];
    devices: HostGpuDevice[];
    reason: string;
  };
  devices: HostDeviceCandidate[];
  warning: string;
}

// MUCLI_CONTAINER_HARDWARE_V1
export interface ContainerCreateOptions {
  source?: 'new' | 'existing';
  existingContainer?: string;
  containerName: string;
  templateName?: string;
  dockerfile?: string;
  mounts?: ContainerMount[];
  gpuRequest?: string;
  devices?: ContainerDevice[];
  egressAllow?: string[];
  egressDeny?: string[];
}

export interface ContainerCreationStatus {
  name: string;
  state: 'idle' | 'running' | 'ready' | 'error';
  stage: string;
  message: string;
  detail?: string;
  logs?: Array<{ seq: number; stream: string; text: string }>;
}

export interface ContainerDefaultsResponse {
  dockerfile: string;
  egress_allow: string[];
  egress_deny: string[];
  hardware?: ContainerHardwareCapabilities;
}

// Types
export interface SessionSummary {
  name: string;
  is_current: boolean;
  is_loaded: boolean;
  is_busy: boolean;
  modified_at: string;
  modified_unix: number;
  session_type?: SessionType;
  container_name?: string | null;
}

export interface SessionListResponse {
  current: string | null;
  active: boolean;
  loaded: string[];
  busy: string[];
  sessions: SessionSummary[];
}

export interface SessionHistoryTurn {
  index: number;
  role: string;
  parts: Array<Record<string, unknown>>;
}

export interface SessionHistoryResponse {
  name: string;
  turns: SessionHistoryTurn[];
  total_turns?: number;
  start_index?: number;
  has_more?: boolean;
  window_end?: number;
}

export interface SessionRequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export interface SessionHistoryOptions extends SessionRequestOptions {
  limitTurns?: number;
  artifactLimit?: number;
  beforeIndex?: number;
  /** Minimum number of user-prompt checkpoints to include when scanning backward. */
  checkpointCount?: number;
  // Round-44 F6: forward pagination — reload evicted newer pages when the
  // user scrolls back down. Server caps the forward window at limit_turns.
  afterIndex?: number;
}

export interface CreateSessionOptions {
  sessionType?: SessionType;
  ollamaMode?: 'local' | 'cloud';
  ollamaHost?: string;
  ollamaApiKey?: string;
  container?: ContainerCreateOptions;
}

export interface WorkspaceSuggestionResponse {
  query: string;
  resolved_path: string;
  exists: boolean;
  suggestions: string[];
}

export interface WorkspaceDetailsResponse {
  name: string;
  workspaces: string[];
}

// API
export const sessionsApi = {
  getContainerDefaults: () => api.get<ContainerDefaultsResponse>('/api/container-defaults'),
  list: (options?: SessionRequestOptions) =>
    api.get<SessionListResponse>('/api/sessions', options),
  getActive: (sessionName?: string, options?: SessionRequestOptions) =>
    api.get<Record<string, unknown>>('/api/sessions/active', {
      ...options,
      query: { session_name: sessionName },
    }),
  getHistory: (sessionName?: string, options?: SessionHistoryOptions) =>
    api.get<SessionHistoryResponse>('/api/sessions/current/history', {
      signal: options?.signal,
      timeoutMs: options?.timeoutMs,
      query: {
        session_name: sessionName,
        limit_turns: options?.limitTurns,
        artifact_limit: options?.artifactLimit,
        before_index: options?.beforeIndex,
        after_index: options?.afterIndex,
        checkpoint_count: options?.checkpointCount,
      },
    }),
  suggestWorkspaces: (path: string, limit: number = 12) =>
    api.get<WorkspaceSuggestionResponse>('/api/sessions/workspaces/suggest', { query: { path, limit } }),
  getWorkspace: (name: string) =>
    api.get<WorkspaceDetailsResponse>(`/api/sessions/${encodeURIComponent(name)}/workspace`),
  updateWorkspace: (name: string, workspaces: string[]) =>
    api.put<WorkspaceDetailsResponse>(`/api/sessions/${encodeURIComponent(name)}/workspace`, { workspaces }),
  create: (name: string, provider: string, model: string, workspace?: string, options?: CreateSessionOptions) =>
    api.post<Record<string, unknown>>('/api/sessions', {
      name,
      provider,
      model,
      activate: true,
      session_type: options?.sessionType || 'workspace',
      workspace,
      ollama_mode: options?.ollamaMode,
      ollama_host: options?.ollamaHost,
      ollama_api_key: options?.ollamaApiKey,
      background_container: options?.sessionType === 'container',
      container_source: options?.container?.source || 'new',
      existing_container: options?.container?.existingContainer,
      container_name: options?.container?.containerName,
      template_name: options?.container?.templateName,
      dockerfile: options?.container?.dockerfile,
      mounts: options?.container?.mounts,
      gpu_request: options?.container?.gpuRequest,
      devices: options?.container?.devices,
      egress_allow: options?.container?.egressAllow,
      egress_deny: options?.container?.egressDeny,
    }),
  getContainerCreationStatus: (name: string, after: number = 0) =>
    api.get<ContainerCreationStatus>(`/api/sessions/creation-status/${encodeURIComponent(name)}`, { query: { after } }),
  load: (name: string, provider?: string, model?: string, options?: SessionRequestOptions) =>
    api.post<Record<string, unknown>>(
      `/api/sessions/${encodeURIComponent(name)}/load`,
      { provider, model },
      options,
    ),
  focus: (name: string, options?: SessionRequestOptions) =>
    api.post<Record<string, unknown>>(
      `/api/sessions/${encodeURIComponent(name)}/focus`,
      undefined,
      options,
    ),
  unloadActive: () => api.delete<void>('/api/sessions/active'),
  detachActive: () => api.post<Record<string, unknown>>('/api/sessions/active/detach'),
  unload: (name: string) => api.post<Record<string, unknown>>(`/api/sessions/${encodeURIComponent(name)}/unload`),
  delete: (name: string) => api.delete<void>(`/api/sessions/${encodeURIComponent(name)}`),
  getContainer: (name: string) =>
    api.get<Record<string, unknown>>(`/api/sessions/${encodeURIComponent(name)}/container`),
  addContainerMount: (name: string, mount: ContainerMount) =>
    api.post<Record<string, unknown>>(`/api/sessions/${encodeURIComponent(name)}/container/mount`, mount),
};
