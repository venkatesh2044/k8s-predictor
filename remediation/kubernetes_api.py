from kubernetes import client, config
import logging
import time

class KubernetesClient:
    """
    Helper class for interacting with the Kubernetes API
    """
    def __init__(self, use_kube_config=False, in_cluster=True):
        self.logger = logging.getLogger(__name__)
        
        # Initialize Kubernetes client
        try:
            if use_kube_config:
                config.load_kube_config()
            elif in_cluster:
                config.load_incluster_config()
            
            self.core_api = client.CoreV1Api()
            self.apps_api = client.AppsV1Api()
            self.batch_api = client.BatchV1Api()
            self.logger.info("Successfully initialized Kubernetes client")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    def get_nodes(self):
        """
        Get all nodes in the cluster
        
        Returns:
            list: List of nodes
        """
        try:
            return self.core_api.list_node().items
        except Exception as e:
            self.logger.error(f"Failed to get nodes: {e}")
            return []
    
    def get_pods(self, namespace=None):
        """
        Get all pods in the cluster or in a specific namespace
        
        Args:
            namespace: Namespace (if None, get pods from all namespaces)
        
        Returns:
            list: List of pods
        """
        try:
            if namespace:
                return self.core_api.list_namespaced_pod(namespace=namespace).items
            else:
                return self.core_api.list_pod_for_all_namespaces().items
        except Exception as e:
            self.logger.error(f"Failed to get pods: {e}")
            return []
    
    def get_deployments(self, namespace=None):
        """
        Get all deployments in the cluster or in a specific namespace
        
        Args:
            namespace: Namespace (if None, get deployments from all namespaces)
        
        Returns:
            list: List of deployments
        """
        try:
            if namespace:
                return self.apps_api.list_namespaced_deployment(namespace=namespace).items
            else:
                return self.apps_api.list_deployment_for_all_namespaces().items
        except Exception as e:
            self.logger.error(f"Failed to get deployments: {e}")
            return []
    
    def get_services(self, namespace=None):
        """
        Get all services in the cluster or in a specific namespace
        
        Args:
            namespace: Namespace (if None, get services from all namespaces)
        
        Returns:
            list: List of services
        """
        try:
            if namespace:
                return self.core_api.list_namespaced_service(namespace=namespace).items
            else:
                return self.core_api.list_service_for_all_namespaces().items
        except Exception as e:
            self.logger.error(f"Failed to get services: {e}")
            return []
    
    def get_configmaps(self, namespace=None):
        """
        Get all configmaps in the cluster or in a specific namespace
        
        Args:
            namespace: Namespace (if None, get configmaps from all namespaces)
        
        Returns:
            list: List of configmaps
        """
        try:
            if namespace:
                return self.core_api.list_namespaced_config_map(namespace=namespace).items
            else:
                return self.core_api.list_config_map_for_all_namespaces().items
        except Exception as e:
            self.logger.error(f"Failed to get configmaps: {e}")
            return []
    
    def get_statefulsets(self, namespace=None):
        """
        Get all statefulsets in the cluster or in a specific namespace
        
        Args:
            namespace: Namespace (if None, get statefulsets from all namespaces)
        
        Returns:
            list: List of statefulsets
        """
        try:
            if namespace:
                return self.apps_api.list_namespaced_stateful_set(namespace=namespace).items
            else:
                return self.apps_api.list_stateful_set_for_all_namespaces().items
        except Exception as e:
            self.logger.error(f"Failed to get statefulsets: {e}")
            return []
    
    def get_daemonsets(self, namespace=None):
        """
        Get all daemonsets in the cluster or in a specific namespace
        
        Args:
            namespace: Namespace (if None, get daemonsets from all namespaces)
        
        Returns:
            list: List of daemonsets
        """
        try:
            if namespace:
                return self.apps_api.list_namespaced_daemon_set(namespace=namespace).items
            else:
                return self.apps_api.list_daemon_set_for_all_namespaces().items
        except Exception as e:
            self.logger.error(f"Failed to get daemonsets: {e}")
            return []
    
    def get_jobs(self, namespace=None):
        """
        Get all jobs in the cluster or in a specific namespace
        
        Args:
            namespace: Namespace (if None, get jobs from all namespaces)
        
        Returns:
            list: List of jobs
        """
        try:
            if namespace:
                return self.batch_api.list_namespaced_job(namespace=namespace).items
            else:
                return self.batch_api.list_job_for_all_namespaces().items
        except Exception as e:
            self.logger.error(f"Failed to get jobs: {e}")
            return []
    
    def create_job(self, namespace, job_definition):
        """
        Create a job in a namespace
        
        Args:
            namespace: Namespace
            job_definition: Job definition
        
        Returns:
            dict: Job object
        """
        try:
            return self.batch_api.create_namespaced_job(
                namespace=namespace,
                body=job_definition
            )
        except Exception as e:
            self.logger.error(f"Failed to create job: {e}")
            return None
    
    def delete_job(self, namespace, job_name):
        """
        Delete a job in a namespace
        
        Args:
            namespace: Namespace
            job_name: Job name
        
        Returns:
            dict: Response object
        """
        try:
            return self.batch_api.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=0
                )
            )
        except Exception as e:
            self.logger.error(f"Failed to delete job: {e}")
            return None
    
    def get_pod_logs(self, namespace, pod_name, container=None, tail_lines=100):
        """
        Get logs from a pod
        
        Args:
            namespace: Namespace
            pod_name: Pod name
            container: Container name (if None, get logs from the first container)
            tail_lines: Number of lines to return
        
        Returns:
            str: Pod logs
        """
        try:
            return self.core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                container=container,
                tail_lines=tail_lines
            )
        except Exception as e:
            self.logger.error(f"Failed to get pod logs: {e}")
            return ""
    
    def wait_for_pod_status(self, namespace, pod_name, status, timeout=60):
        """
        Wait for a pod to reach a specific status
        
        Args:
            namespace: Namespace
            pod_name: Pod name
            status: Status to wait for
            timeout: Timeout in seconds
        
        Returns:
            bool: True if pod reached the status, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                pod = self.core_api.read_namespaced_pod(
                    name=pod_name,
                    namespace=namespace
                )
                if pod.status.phase == status:
                    return True
            except Exception as e:
                self.logger.error(f"Error while waiting for pod {pod_name}: {e}")
                return False
            
            time.sleep(1)
        
        return False