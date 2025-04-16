import time
import pandas as pd
import numpy as np
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
import logging

class DataCollector:
    """
    Collects metrics from Kubernetes API and Prometheus
    """
    def __init__(self, prometheus_url=None, use_kube_config=False, in_cluster=True):
        """
        Initialize the data collector
        
        Args:
            prometheus_url: URL to Prometheus server (optional)
            use_kube_config: Whether to use local kube config
            in_cluster: Whether running inside Kubernetes cluster
        """
        self.logger = logging.getLogger(__name__)
        self.prometheus_url = prometheus_url
        
        # Initialize Kubernetes client
        try:
            if use_kube_config:
                config.load_kube_config()
            elif in_cluster:
                config.load_incluster_config()
            
            self.core_api = client.CoreV1Api()
            self.apps_api = client.AppsV1Api()
            self.logger.info("Successfully initialized Kubernetes client")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
        
        # Initialize Prometheus client if URL provided
        if prometheus_url:
            try:
                self.prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
                self.logger.info(f"Successfully connected to Prometheus at {prometheus_url}")
            except Exception as e:
                self.logger.error(f"Failed to connect to Prometheus: {e}")
                self.prom = None
        else:
            self.prom = None
    
    def get_node_metrics(self):
        """
        Get metrics for all nodes in the cluster
        
        Returns:
            DataFrame with node metrics
        """
        try:
            nodes = self.core_api.list_node().items
            node_metrics = []
            
            for node in nodes:
                node_name = node.metadata.name
                node_status = node.status
                
                # Basic node info
                node_info = {
                    'timestamp': time.time(),
                    'node_name': node_name,
                    'node_condition': self._get_node_condition(node_status),
                    'pod_count': self._count_pods_on_node(node_name)
                }
                
                # Get metrics from Prometheus if available
                if self.prom:
                    cpu_usage = self._get_prometheus_metric(f'node_cpu_usage_percent{{node="{node_name}"}}')
                    memory_usage = self._get_prometheus_metric(f'node_memory_usage_percent{{node="{node_name}"}}')
                    disk_usage = self._get_prometheus_metric(f'node_disk_usage_percent{{node="{node_name}"}}')
                    network_load = self._get_prometheus_metric(f'node_network_load_mbps{{node="{node_name}"}}')
                    
                    node_info.update({
                        'cpu_usage_percent': cpu_usage if cpu_usage is not None else self._generate_mock_metric(30, 70),
                        'memory_usage_percent': memory_usage if memory_usage is not None else self._generate_mock_metric(40, 60),
                        'disk_usage_percent': disk_usage if disk_usage is not None else self._generate_mock_metric(50, 30),
                        'network_load_mbps': network_load if network_load is not None else self._generate_mock_metric(20, 10),
                    })
                else:
                    # Generate mock metrics if Prometheus is not available
                    node_info.update({
                        'cpu_usage_percent': self._generate_mock_metric(30, 70),
                        'memory_usage_percent': self._generate_mock_metric(40, 60),
                        'disk_usage_percent': self._generate_mock_metric(50, 30),
                        'network_load_mbps': self._generate_mock_metric(20, 10),
                    })
                
                # Get capacity and allocatable resources
                allocatable = node_status.allocatable
                capacity = node_status.capacity
                
                node_info.update({
                    'available_memory_bytes': int(allocatable.get('memory', '0').replace('Ki', '')) * 1024,
                    'total_memory_bytes': int(capacity.get('memory', '0').replace('Ki', '')) * 1024,
                    'available_disk_bytes': int(allocatable.get('ephemeral-storage', '0').replace('Ki', '')) * 1024,
                    'total_disk_bytes': int(capacity.get('ephemeral-storage', '0').replace('Ki', '')) * 1024,
                })
                
                # Additional metrics
                node_info.update({
                    'network_errors': self._generate_mock_metric(1, 2),
                    'network_drops': self._generate_mock_metric(1, 2),
                    'api_server_latency_ms': self._generate_mock_metric(5, 10),
                    'etcd_latency_ms': self._generate_mock_metric(3, 5),
                    'response_time_ms': self._generate_mock_metric(10, 15)
                })
                
                node_metrics.append(node_info)
            
            return pd.DataFrame(node_metrics)
        
        except Exception as e:
            self.logger.error(f"Error collecting node metrics: {e}")
            return pd.DataFrame()
    
    def get_pod_metrics(self):
        """
        Get metrics for all pods in the cluster
        
        Returns:
            DataFrame with pod metrics
        """
        try:
            pods = self.core_api.list_pod_for_all_namespaces().items
            pod_metrics = []
            
            for pod in pods:
                pod_name = pod.metadata.name
                namespace = pod.metadata.namespace
                node_name = pod.spec.node_name if pod.spec.node_name else "unknown"
                status_phase = pod.status.phase
                
                # Get container statuses
                container_statuses = []
                restart_count = 0
                
                if pod.status.container_statuses:
                    for container in pod.status.container_statuses:
                        container_statuses.append(
                            'running' if container.ready else 'not_ready'
                        )
                        restart_count += container.restart_count
                
                container_status = ','.join(container_statuses) if container_statuses else 'unknown'
                
                # Basic pod info
                pod_info = {
                    'timestamp': time.time(),
                    'pod_name': pod_name,
                    'pod_namespace': namespace,
                    'node_name': node_name,
                    'pod_status': status_phase,
                    'container_status': container_status,
                    'restart_count': restart_count
                }
                
                # Get metrics from Prometheus if available
                if self.prom:
                    cpu_usage = self._get_prometheus_metric(f'pod_cpu_usage_percent{{pod="{pod_name}", namespace="{namespace}"}}')
                    memory_usage = self._get_prometheus_metric(f'pod_memory_usage_percent{{pod="{pod_name}", namespace="{namespace}"}}')
                    network_usage = self._get_prometheus_metric(f'pod_network_mbps{{pod="{pod_name}", namespace="{namespace}"}}')
                    
                    pod_info.update({
                        'pod_cpu_usage_percent': cpu_usage if cpu_usage is not None else self._generate_mock_metric(20, 30),
                        'pod_memory_usage_percent': memory_usage if memory_usage is not None else self._generate_mock_metric(30, 25),
                        'pod_network_mbps': network_usage if network_usage is not None else self._generate_mock_metric(5, 3),
                    })
                else:
                    # Generate mock metrics if Prometheus is not available
                    pod_info.update({
                        'pod_cpu_usage_percent': self._generate_mock_metric(20, 30),
                        'pod_memory_usage_percent': self._generate_mock_metric(30, 25),
                        'pod_network_mbps': self._generate_mock_metric(5, 3),
                    })
                
                # Calculate issue label for training
                # This is a simplistic approach - in reality this would be based on historical data
                issue_label = 'none'
                if restart_count > 5:
                    issue_label = 'frequent_restarts'
                elif status_phase == 'Failed' or 'CrashLoopBackOff' in container_status:
                    issue_label = 'pod_crash'
                elif pod_info['pod_cpu_usage_percent'] > 85:
                    issue_label = 'resource_exhaustion_cpu'
                elif pod_info['pod_memory_usage_percent'] > 85:
                    issue_label = 'resource_exhaustion_memory'
                
                pod_info['issue_label'] = issue_label
                pod_metrics.append(pod_info)
            
            return pd.DataFrame(pod_metrics)
        
        except Exception as e:
            self.logger.error(f"Error collecting pod metrics: {e}")
            return pd.DataFrame()
    
    def collect_all_metrics(self):
        """
        Collect all metrics from the cluster
        
        Returns:
            Tuple of (node_metrics_df, pod_metrics_df)
        """
        node_metrics = self.get_node_metrics()
        pod_metrics = self.get_pod_metrics()
        
        self.logger.info(f"Collected metrics for {len(node_metrics)} nodes and {len(pod_metrics)} pods")
        
        return node_metrics, pod_metrics
    
    def _get_node_condition(self, node_status):
        """Get the condition of a node"""
        for condition in node_status.conditions:
            if condition.type == 'Ready':
                return condition.status
        return 'Unknown'
    
    def _count_pods_on_node(self, node_name):
        """Count pods running on a specific node"""
        pods = self.core_api.list_pod_for_all_namespaces(field_selector=f'spec.nodeName={node_name}')
        return len(pods.items)
    
    def _get_prometheus_metric(self, query):
        """Get a single metric from Prometheus"""
        try:
            if self.prom:
                result = self.prom.custom_query(query=query)
                if result and len(result) > 0 and 'value' in result[0]:
                    return float(result[0]['value'][1])
            return None
        except Exception as e:
            self.logger.warning(f"Error getting Prometheus metric {query}: {e}")
            return None
    
    def _generate_mock_metric(self, base, variance):
        """Generate a mock metric value for testing without Prometheus"""
        return round(base + variance * np.random.random(), 2)