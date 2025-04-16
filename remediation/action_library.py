import logging
import time
from kubernetes import client, config

class RemediationActions:
    """
    Library of remediation actions for Kubernetes issues
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
            self.logger.info("Successfully initialized Kubernetes client for remediation")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    def restart_pod(self, namespace, pod_name):
        """
        Restart a pod by deleting it (it will be recreated by its controller)
        
        Args:
            namespace: Pod namespace
            pod_name: Pod name
        
        Returns:
            dict: Result of the action
        """
        try:
            self.logger.info(f"Restarting pod {pod_name} in namespace {namespace}")
            self.core_api.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace,
                body=client.V1DeleteOptions()
            )
            # Wait to ensure pod deletion has started
            time.sleep(2)
            return {
                "success": True,
                "action": "restart_pod",
                "target": f"{namespace}/{pod_name}",
                "message": f"Successfully triggered restart of pod {pod_name}"
            }
        except Exception as e:
            self.logger.error(f"Failed to restart pod {pod_name}: {e}")
            return {
                "success": False,
                "action": "restart_pod",
                "target": f"{namespace}/{pod_name}",
                "message": f"Failed to restart pod: {str(e)}"
            }
    
    def scale_deployment(self, namespace, deployment_name, replicas):
        """
        Scale a deployment to a specified number of replicas
        
        Args:
            namespace: Deployment namespace
            deployment_name: Deployment name
            replicas: Target number of replicas
        
        Returns:
            dict: Result of the action
        """
        try:
            self.logger.info(f"Scaling deployment {deployment_name} to {replicas} replicas in namespace {namespace}")
            
            # Get current scale
            current_scale = self.apps_api.read_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update scale
            current_scale.spec.replicas = replicas
            self.apps_api.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body=current_scale
            )
            
            return {
                "success": True,
                "action": "scale_deployment",
                "target": f"{namespace}/{deployment_name}",
                "message": f"Successfully scaled deployment {deployment_name} to {replicas} replicas"
            }
        except Exception as e:
            self.logger.error(f"Failed to scale deployment {deployment_name}: {e}")
            return {
                "success": False,
                "action": "scale_deployment",
                "target": f"{namespace}/{deployment_name}",
                "message": f"Failed to scale deployment: {str(e)}"
            }
    
    def cordon_node(self, node_name):
        """
        Cordon a node to prevent new pods from being scheduled
        
        Args:
            node_name: Node name
        
        Returns:
            dict: Result of the action
        """
        try:
            self.logger.info(f"Cordoning node {node_name}")
            
            # Get current node
            node = self.core_api.read_node(name=node_name)
            
            # Set unschedulable
            node.spec.unschedulable = True
            self.core_api.patch_node(name=node_name, body=node)
            
            return {
                "success": True,
                "action": "cordon_node",
                "target": node_name,
                "message": f"Successfully cordoned node {node_name}"
            }
        except Exception as e:
            self.logger.error(f"Failed to cordon node {node_name}: {e}")
            return {
                "success": False,
                "action": "cordon_node",
                "target": node_name,
                "message": f"Failed to cordon node: {str(e)}"
            }
    
    def drain_node(self, node_name, delete_local_data=False, ignore_daemonsets=True, timeout_seconds=300):
        """
        Drain a node (must be cordoned first)
        
        Args:
            node_name: Node name
            delete_local_data: Whether to delete local data
            ignore_daemonsets: Whether to ignore DaemonSets
            timeout_seconds: Timeout in seconds
        
        Returns:
            dict: Result of the action
        """
        try:
            self.logger.info(f"Draining node {node_name}")
            
            # Get pods on the node
            field_selector = f"spec.nodeName={node_name}"
            pods = self.core_api.list_pod_for_all_namespaces(field_selector=field_selector)
            
            # Cordon the node first if not already cordoned
            try:
                node = self.core_api.read_node(name=node_name)
                if not node.spec.unschedulable:
                    self.cordon_node(node_name)
            except Exception as e:
                self.logger.error(f"Failed to cordon node before draining: {e}")
            
            # Filter pods to evict
            pods_to_evict = []
            for pod in pods.items:
                # Skip DaemonSets if ignore_daemonsets is True
                if ignore_daemonsets:
                    controller_kind = None
                    for owner_ref in pod.metadata.owner_references or []:
                        if owner_ref.controller and owner_ref.kind == "DaemonSet":
                            controller_kind = "DaemonSet"
                            break
                    if controller_kind == "DaemonSet":
                        continue
                
                # Skip mirror pods
                if pod.metadata.annotations and "kubernetes.io/config.mirror" in pod.metadata.annotations:
                    continue
                
                # Add to eviction list
                pods_to_evict.append((pod.metadata.name, pod.metadata.namespace))
            
            # Evict pods
            eviction_success = True
            failed_pods = []
            
            for pod_name, namespace in pods_to_evict:
                try:
                    eviction = client.V1Eviction(
                        metadata=client.V1ObjectMeta(
                            name=pod_name,
                            namespace=namespace
                        )
                    )
                    self.core_api.create_namespaced_pod_eviction(
                        name=pod_name,
                        namespace=namespace,
                        body=eviction
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to evict pod {namespace}/{pod_name}: {e}")
                    eviction_success = False
                    failed_pods.append(f"{namespace}/{pod_name}")
            
            return {
                "success": eviction_success,
                "action": "drain_node",
                "target": node_name,
                "message": f"Node drain {'completed' if eviction_success else 'partially completed'} for {node_name}",
                "failed_pods": failed_pods if not eviction_success else []
            }
        except Exception as e:
            self.logger.error(f"Failed to drain node {node_name}: {e}")
            return {
                "success": False,
                "action": "drain_node",
                "target": node_name,
                "message": f"Failed to drain node: {str(e)}"
            }
    
    def evict_pod(self, namespace, pod_name):
        """
        Evict a pod
        
        Args:
            namespace: Pod namespace
            pod_name: Pod name
        
        Returns:
            dict: Result of the action
        """
        try:
            self.logger.info(f"Evicting pod {pod_name} in namespace {namespace}")
            
            eviction = client.V1Eviction(
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    namespace=namespace
                )
            )
            self.core_api.create_namespaced_pod_eviction(
                name=pod_name,
                namespace=namespace,
                body=eviction
            )
            
            return {
                "success": True,
                "action": "evict_pod",
                "target": f"{namespace}/{pod_name}",
                "message": f"Successfully evicted pod {pod_name}"
            }
        except Exception as e:
            self.logger.error(f"Failed to evict pod {pod_name}: {e}")
            return {
                "success": False,
                "action": "evict_pod",
                "target": f"{namespace}/{pod_name}",
                "message": f"Failed to evict pod: {str(e)}"
            }
    
    def update_resource_limits(self, namespace, deployment_name, container_name=None, cpu_limit=None, memory_limit=None):
        """
        Update resource limits for a deployment
        
        Args:
            namespace: Deployment namespace
            deployment_name: Deployment name
            container_name: Container name (if None, update all containers)
            cpu_limit: New CPU limit
            memory_limit: New memory limit
        
        Returns:
            dict: Result of the action
        """
        try:
            self.logger.info(f"Updating resource limits for deployment {deployment_name} in namespace {namespace}")
            
            # Get current deployment
            deployment = self.apps_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update resource limits
            containers = deployment.spec.template.spec.containers
            updated = False
            
            for container in containers:
                if container_name is None or container.name == container_name:
                    if not container.resources:
                        container.resources = client.V1ResourceRequirements(limits={}, requests={})
                    
                    if not container.resources.limits:
                        container.resources.limits = {}
                    
                    if cpu_limit is not None:
                        container.resources.limits['cpu'] = cpu_limit
                        updated = True
                    
                    if memory_limit is not None:
                        container.resources.limits['memory'] = memory_limit
                        updated = True
            
            # Update deployment if changes were made
            if updated:
                self.apps_api.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment
                )
                
                return {
                    "success": True,
                    "action": "update_resource_limits",
                    "target": f"{namespace}/{deployment_name}",
                    "message": f"Successfully updated resource limits for deployment {deployment_name}"
                }
            else:
                return {
                    "success": False,
                    "action": "update_resource_limits",
                    "target": f"{namespace}/{deployment_name}",
                    "message": "No changes were made to resource limits"
                }
        except Exception as e:
            self.logger.error(f"Failed to update resource limits for deployment {deployment_name}: {e}")
            return {
                "success": False,
                "action": "update_resource_limits",
                "target": f"{namespace}/{deployment_name}",
                "message": f"Failed to update resource limits: {str(e)}"
            }
    
    def add_toleration(self, namespace, deployment_name, key, operator="Equal", value="", effect="NoSchedule"):
        """
        Add a toleration to a deployment
        
        Args:
            namespace: Deployment namespace
            deployment_name: Deployment name
            key: Toleration key
            operator: Toleration operator
            value: Toleration value
            effect: Toleration effect
        
        Returns:
            dict: Result of the action
        """
        try:
            self.logger.info(f"Adding toleration {key}:{value} to deployment {deployment_name} in namespace {namespace}")
            
            # Get current deployment
            deployment = self.apps_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Create toleration
            toleration = client.V1Toleration(
                key=key,
                operator=operator,
                value=value,
                effect=effect
            )
            
            # Check if toleration already exists
            tolerations = deployment.spec.template.spec.tolerations or []
            for t in tolerations:
                if (t.key == key and t.operator == operator and 
                    t.value == value and t.effect == effect):
                    return {
                        "success": True,
                        "action": "add_toleration",
                        "target": f"{namespace}/{deployment_name}",
                        "message": f"Toleration already exists for deployment {deployment_name}"
                    }
            
            # Add toleration
            tolerations.append(toleration)
            deployment.spec.template.spec.tolerations = tolerations
            
            # Update deployment
            self.apps_api.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            return {
                "success": True,
                "action": "add_toleration",
                "target": f"{namespace}/{deployment_name}",
                "message": f"Successfully added toleration to deployment {deployment_name}"
            }
        except Exception as e:
            self.logger.error(f"Failed to add toleration to deployment {deployment_name}: {e}")
            return {
                "success": False,
                "action": "add_toleration",
                "target": f"{namespace}/{deployment_name}",
                "message": f"Failed to add toleration: {str(e)}"
            }
    
    def taint_node(self, node_name, key, value="", effect="NoSchedule"):
        """
        Add a taint to a node
        
        Args:
            node_name: Node name
            key: Taint key
            value: Taint value
            effect: Taint effect
        
        Returns:
            dict: Result of the action
        """
        try:
            self.logger.info(f"Adding taint {key}:{value} to node {node_name}")
            
            # Get current node
            node = self.core_api.read_node(name=node_name)
            
            # Create taint
            taint = client.V1Taint(
                key=key,
                value=value,
                effect=effect
            )
            
            # Check if taint already exists
            taints = node.spec.taints or []
            for t in taints:
                if t.key == key and t.value == value and t.effect == effect:
                    return {
                        "success": True,
                        "action": "taint_node",
                        "target": node_name,
                        "message": f"Taint already exists for node {node_name}"
                    }
            
            # Add taint
            taints.append(taint)
            node.spec.taints = taints
            
            # Update node
            self.core_api.patch_node(name=node_name, body=node)
            
            return {
                "success": True,
                "action": "taint_node",
                "target": node_name,
                "message": f"Successfully added taint to node {node_name}"
            }
        except Exception as e:
            self.logger.error(f"Failed to add taint to node {node_name}: {e}")
            return {
                "success": False,
                "action": "taint_node",
                "target": node_name,
                "message": f"Failed to add taint: {str(e)}"
            }