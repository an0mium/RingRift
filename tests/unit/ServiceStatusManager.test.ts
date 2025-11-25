/**
 * Tests for ServiceStatusManager - graceful degradation patterns
 */

import {
  ServiceStatusManager,
  DegradationLevel,
  ServiceHealthStatus,
  ServiceName,
  getServiceStatusManager,
  initServiceStatusManager,
  resetServiceStatusManager,
} from '../../src/server/services/ServiceStatusManager';

describe('ServiceStatusManager', () => {
  let statusManager: ServiceStatusManager;

  beforeEach(() => {
    // Reset singleton and create fresh instance
    resetServiceStatusManager();
    statusManager = new ServiceStatusManager();
  });

  afterEach(() => {
    statusManager.destroy();
  });

  describe('Initialization', () => {
    it('should initialize with all services in unknown status', () => {
      const systemStatus = statusManager.getSystemStatus();

      expect(systemStatus.services.database.status).toBe('unknown');
      expect(systemStatus.services.redis.status).toBe('unknown');
      expect(systemStatus.services.aiService.status).toBe('unknown');
    });

    it('should initialize with FULL degradation level', () => {
      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.FULL);
    });
  });

  describe('Service Status Updates', () => {
    it('should update service status correctly', () => {
      statusManager.updateServiceStatus('database', 'healthy');

      const dbStatus = statusManager.getServiceStatus('database');
      expect(dbStatus?.status).toBe('healthy');
      expect(dbStatus?.failureCount).toBe(0);
    });

    it('should track failure count on status changes', () => {
      statusManager.updateServiceStatus('aiService', 'healthy');
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Connection refused');
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Connection refused');

      const aiStatus = statusManager.getServiceStatus('aiService');
      expect(aiStatus?.failureCount).toBe(1); // Only counts status changes
    });

    it('should record lastHealthy timestamp', () => {
      statusManager.updateServiceStatus('database', 'healthy');

      const dbStatus = statusManager.getServiceStatus('database');
      expect(dbStatus?.lastHealthy).toBeDefined();
    });

    it('should store error message for unhealthy services', () => {
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Connection timeout');

      const redisStatus = statusManager.getServiceStatus('redis');
      expect(redisStatus?.error).toBe('Connection timeout');
    });

    it('should clear error message when service becomes healthy', () => {
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Connection timeout');
      statusManager.updateServiceStatus('redis', 'healthy');

      const redisStatus = statusManager.getServiceStatus('redis');
      expect(redisStatus?.error).toBeUndefined();
    });
  });

  describe('Degradation Level Computation', () => {
    it('should be FULL when all services are healthy', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'healthy');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.FULL);
    });

    it('should be DEGRADED when AI service is unhealthy', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Service unavailable');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.DEGRADED);
    });

    it('should be DEGRADED when Redis is unhealthy', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Connection refused');
      statusManager.updateServiceStatus('aiService', 'healthy');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.DEGRADED);
    });

    it('should be DEGRADED when database is degraded', () => {
      statusManager.updateServiceStatus('database', 'degraded', 'High latency');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'healthy');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.DEGRADED);
    });

    it('should be MINIMAL when AI and Redis are both down', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Connection refused');
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Service unavailable');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.MINIMAL);
    });

    it('should be MINIMAL when database is degraded and a non-critical service is down', () => {
      statusManager.updateServiceStatus('database', 'degraded', 'High latency');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Service unavailable');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.MINIMAL);
    });

    it('should be OFFLINE when database is unhealthy', () => {
      statusManager.updateServiceStatus('database', 'unhealthy', 'Connection refused');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'healthy');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.OFFLINE);
    });

    it('should be OFFLINE when database is unhealthy regardless of other services', () => {
      statusManager.updateServiceStatus('database', 'unhealthy', 'Connection refused');
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Connection refused');
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Service unavailable');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.OFFLINE);
    });
  });

  describe('Event Emission', () => {
    it('should emit statusChange event when service status changes', (done) => {
      statusManager.on('statusChange', (service, oldStatus, newStatus) => {
        expect(service).toBe('database');
        expect(oldStatus).toBe('unknown');
        expect(newStatus).toBe('healthy');
        done();
      });

      statusManager.updateServiceStatus('database', 'healthy');
    });

    it('should emit degradationLevelChange event when level changes', (done) => {
      // First set all healthy
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'healthy');

      statusManager.on('degradationLevelChange', (oldLevel, newLevel) => {
        expect(oldLevel).toBe(DegradationLevel.FULL);
        expect(newLevel).toBe(DegradationLevel.DEGRADED);
        done();
      });

      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Service down');
    });

    it('should emit serviceRecovered event when service recovers', (done) => {
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Service down');

      statusManager.on('serviceRecovered', (service) => {
        expect(service).toBe('aiService');
        done();
      });

      statusManager.updateServiceStatus('aiService', 'healthy');
    });

    it('should emit serviceDown event when service becomes unavailable', (done) => {
      statusManager.updateServiceStatus('redis', 'healthy');

      statusManager.on('serviceDown', (service, error) => {
        expect(service).toBe('redis');
        expect(error).toBe('Connection lost');
        done();
      });

      statusManager.updateServiceStatus('redis', 'unhealthy', 'Connection lost');
    });
  });

  describe('Degradation Headers', () => {
    it('should return empty headers when all services are healthy', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'healthy');

      const headers = statusManager.getDegradationHeaders();
      expect(Object.keys(headers).length).toBe(0);
    });

    it('should return X-Service-Status header when degraded', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Service down');

      const headers = statusManager.getDegradationHeaders();
      expect(headers['X-Service-Status']).toBe('degraded');
    });

    it('should return X-Degraded-Services header listing degraded services', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Connection refused');
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Service down');

      const headers = statusManager.getDegradationHeaders();
      expect(headers['X-Degraded-Services']).toContain('redis');
      expect(headers['X-Degraded-Services']).toContain('aiService');
    });
  });

  describe('Helper Methods', () => {
    it('should correctly report isDegraded', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'healthy');

      expect(statusManager.isDegraded()).toBe(false);

      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Down');

      expect(statusManager.isDegraded()).toBe(true);
    });

    it('should correctly report isServiceHealthy', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Down');

      expect(statusManager.isServiceHealthy('database')).toBe(true);
      expect(statusManager.isServiceHealthy('redis')).toBe(false);
    });

    it('should list degraded services', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Down');
      statusManager.updateServiceStatus('aiService', 'degraded', 'Slow');

      const degraded = statusManager.getDegradedServices();
      expect(degraded).toContain('redis');
      expect(degraded).toContain('aiService');
      expect(degraded).not.toContain('database');
    });
  });

  describe('Health Check Callbacks', () => {
    it('should register and execute health check callbacks', async () => {
      const mockCallback = jest.fn().mockResolvedValue({
        status: 'healthy' as ServiceHealthStatus,
        latencyMs: 50,
      });

      statusManager.registerHealthCheck('database', mockCallback);
      await statusManager.runHealthChecks();

      expect(mockCallback).toHaveBeenCalled();
    });

    it('should update service status from health check result', async () => {
      statusManager.registerHealthCheck('database', async () => ({
        status: 'healthy',
        latencyMs: 25,
      }));

      await statusManager.runHealthChecks();

      const dbStatus = statusManager.getServiceStatus('database');
      expect(dbStatus?.status).toBe('healthy');
      expect(dbStatus?.latencyMs).toBe(25);
    });

    it('should handle failing health check callbacks', async () => {
      statusManager.registerHealthCheck('redis', async () => {
        throw new Error('Connection failed');
      });

      await statusManager.runHealthChecks();

      const redisStatus = statusManager.getServiceStatus('redis');
      expect(redisStatus?.status).toBe('unhealthy');
      expect(redisStatus?.error).toBe('Connection failed');
    });
  });

  describe('Polling', () => {
    it('should start and stop polling', () => {
      const manager = new ServiceStatusManager({
        enablePolling: true,
        pollingIntervalMs: 1000,
      });

      manager.startPolling();
      // Just verify it doesn't throw
      manager.stopPolling();
      manager.destroy();
    });
  });

  describe('Singleton Pattern', () => {
    it('should return singleton instance', () => {
      resetServiceStatusManager();
      const instance1 = getServiceStatusManager();
      const instance2 = getServiceStatusManager();

      expect(instance1).toBe(instance2);
    });

    it('should create new instance with initServiceStatusManager', () => {
      resetServiceStatusManager();
      const instance1 = getServiceStatusManager();
      const instance2 = initServiceStatusManager({ pollingIntervalMs: 5000 });

      expect(instance1).not.toBe(instance2);
    });
  });

  describe('Reset', () => {
    it('should reset all services to unknown status', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Down');

      statusManager.reset();

      const dbStatus = statusManager.getServiceStatus('database');
      const redisStatus = statusManager.getServiceStatus('redis');

      expect(dbStatus?.status).toBe('unknown');
      expect(redisStatus?.status).toBe('unknown');
    });
  });

  describe('Fallback Active Tracking', () => {
    it('should set fallback active flag', () => {
      statusManager.setFallbackActive('aiService', true);

      const aiStatus = statusManager.getServiceStatus('aiService');
      expect(aiStatus?.fallbackActive).toBe(true);
    });

    it('should automatically set fallback active when service becomes unhealthy', () => {
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Down');

      const aiStatus = statusManager.getServiceStatus('aiService');
      expect(aiStatus?.fallbackActive).toBe(true);
    });

    it('should clear fallback active when service becomes healthy', () => {
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Down');
      statusManager.updateServiceStatus('aiService', 'healthy');

      const aiStatus = statusManager.getServiceStatus('aiService');
      expect(aiStatus?.fallbackActive).toBe(false);
    });
  });
});

describe('Degradation Scenarios', () => {
  let statusManager: ServiceStatusManager;

  beforeEach(() => {
    resetServiceStatusManager();
    statusManager = new ServiceStatusManager();
  });

  afterEach(() => {
    statusManager.destroy();
  });

  describe('AI Service Timeout', () => {
    it('should mark system as DEGRADED and indicate fallback', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'degraded', 'Request timed out');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.DEGRADED);
      expect(statusManager.getServiceStatus('aiService')?.fallbackActive).toBe(true);

      const headers = statusManager.getDegradationHeaders();
      expect(headers['X-Service-Status']).toBe('degraded');
      expect(headers['X-Degraded-Services']).toContain('aiService');
    });
  });

  describe('Redis Unavailable', () => {
    it('should mark system as DEGRADED', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Connection refused');
      statusManager.updateServiceStatus('aiService', 'healthy');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.DEGRADED);
    });
  });

  describe('AI + Redis Down', () => {
    it('should mark system as MINIMAL', () => {
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'unhealthy', 'Connection refused');
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Service unavailable');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.MINIMAL);
    });
  });

  describe('Database Down', () => {
    it('should mark system as OFFLINE', () => {
      statusManager.updateServiceStatus('database', 'unhealthy', 'Connection refused');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'healthy');

      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.OFFLINE);
    });
  });

  describe('Recovery Scenario', () => {
    it('should recover from DEGRADED to FULL when AI service comes back', () => {
      const levelChanges: [DegradationLevel, DegradationLevel][] = [];

      // Start with all healthy
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'healthy');

      statusManager.on('degradationLevelChange', (oldLevel, newLevel) => {
        levelChanges.push([oldLevel, newLevel]);
      });

      // AI service goes down
      statusManager.updateServiceStatus('aiService', 'unhealthy', 'Service down');
      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.DEGRADED);

      // AI service recovers
      statusManager.updateServiceStatus('aiService', 'healthy');
      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.FULL);

      // Verify events
      expect(levelChanges).toHaveLength(2);
      expect(levelChanges[0]).toEqual([DegradationLevel.FULL, DegradationLevel.DEGRADED]);
      expect(levelChanges[1]).toEqual([DegradationLevel.DEGRADED, DegradationLevel.FULL]);
    });

    it('should recover from OFFLINE to FULL when database comes back', () => {
      // Start with all healthy
      statusManager.updateServiceStatus('database', 'healthy');
      statusManager.updateServiceStatus('redis', 'healthy');
      statusManager.updateServiceStatus('aiService', 'healthy');

      // Database goes down
      statusManager.updateServiceStatus('database', 'unhealthy', 'Connection lost');
      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.OFFLINE);

      // Database recovers
      statusManager.updateServiceStatus('database', 'healthy');
      expect(statusManager.getDegradationLevel()).toBe(DegradationLevel.FULL);
    });
  });
});