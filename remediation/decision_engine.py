import logging
import pandas as pd
import numpy as np

class RemediationDecisionEngine:
    """
    Decision engine for selecting appropriate remediation actions
    """
    def __init__(self, action_library):
        self.logger = logging.getLogger(__name__)
        self.actions = action_library
        
        # Define thresholds for different issues
        self.thresholds = {
            'pod_crash_probability': 0.7,
            'node_failure_probability': 0.7,
            'network_issue_probability': 0.7,
            'cpu_usage_percent': 85,
            'memory_usage_percent': 85,
            'disk_usage_percent': 85,
            'network_load_mbps': 80
        }
    
    def decide_pod_actions(self, pod_metrics, pod_predictions):
        """
        Decide actions for pod issues
        
        Args:
            pod_metrics: DataFrame with current pod metrics
            pod_predictions: DataFrame with pod predictions
        
        Returns:
            list: List of remediation actions to take
        """
        actions = []
        
        # Combine data
        combined_data = pd.merge(
            pod_metrics,
            pod_predictions,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        for idx, row in combined_data.iterrows():
            pod_name = row.get('pod_name')
            namespace = row.get('pod_namespace', 'default')
            node_name = row.get('node_name', '')
            
            # Check for crash predictions
            if 'crash_probability' in row and row['crash_probability'] > self.thresholds['pod_crash_probability']:
                actions.append({
                    'action': 'restart_pod',
                    'params': {
                        'namespace': namespace,
                        'pod_name': pod_name
                    },
                    'reason': f"High crash probability: {row['crash_probability']:.2f}",
                    'priority': 1
                })
            
            # Check for resource exhaustion
            cpu_warning = row.get('cpu_warning', False)
            memory_warning = row.get('memory_warning', False)
            
            if cpu_warning or memory_warning:
                # Find deployment for this pod
                deployment_name = self._get_deployment_for_pod(pod_name)
                
                if deployment_name:
                    if cpu_warning and memory_warning:
                        actions.append({
                            'action': 'scale_deployment',
                            'params': {
                                'namespace': namespace,
                                'deployment_name': deployment_name,
                                'replicas': '+'  # Increase by 1
                            },
                            'reason': "High CPU and memory usage predicted",
                            'priority': 2
                        })
                    elif cpu_warning:
                        actions.append({
                            'action': 'update_resource_limits',
                            'params': {
                                'namespace': namespace,
                                'deployment_name': deployment_name,
                                'cpu_limit': str(int(row.get('pod_cpu_usage_percent', 50) * 1.5 / 100)) + 'i'
                            },
                            'reason': "High CPU usage predicted",
                            'priority': 3
                        })
                    elif memory_warning:
                        actions.append({
                            'action': 'update_resource_limits',
                            'params': {
                                'namespace': namespace,
                                'deployment_name': deployment_name,
                                'memory_limit': str(int(row.get('pod_memory_usage_percent', 50) * 1.5)) + 'Mi'
                            },
                            'reason': "High memory usage predicted",
                            'priority': 3
                        })
        
        return actions
    
    def decide_node_actions(self, node_metrics, node_predictions):
        """
        Decide actions for node issues
        
        Args:
            node_metrics: DataFrame with current node metrics
            node_predictions: DataFrame with node predictions
        
        Returns:
            list: List of remediation actions to take
        """
        actions = []
        
        # Combine data
        combined_data = pd.merge(
            node_metrics,
            node_predictions,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        for idx, row in combined_data.iterrows():
            node_name = row.get('node_name')
            
            # Check for node failure predictions
            if 'failure_probability' in row and row['failure_probability'] > self.thresholds['node_failure_probability']:
                actions.append({
                    'action': 'cordon_node',
                    'params': {
                        'node_name': node_name
                    },
                    'reason': f"High node failure probability: {row['failure_probability']:.2f}",
                    'priority': 1
                })
                
                actions.append({
                    'action': 'drain_node',
                    'params': {
                        'node_name': node_name,
                        'delete_local_data': False,
                        'ignore_daemonsets': True
                    },
                    'reason': f"High node failure probability: {row['failure_probability']:.2f}",
                    'priority': 2
                })
            
            # Check for resource exhaustion
            cpu_warning = row.get('cpu_warning', False) or (row.get('cpu_usage_percent', 0) > self.thresholds['cpu_usage_percent'])
            memory_warning = row.get('memory_warning', False) or (row.get('memory_usage_percent', 0) > self.thresholds['memory_usage_percent'])
            disk_warning = row.get('disk_warning', False) or (row.get('disk_usage_percent', 0) > self.thresholds['disk_usage_percent'])
            network_warning = row.get('network_warning', False) or (row.get('network_load_mbps', 0) > self.thresholds['network_load_mbps'])
            
            if cpu_warning or memory_warning:
                actions.append({
                    'action': 'taint_node',
                    'params': {
                        'node_name': node_name,
                        'key': 'node.kubernetes.io/high-load',
                        'value': 'true',
                        'effect': 'NoSchedule'
                    },
                    'reason': "High CPU or memory usage predicted",
                    'priority': 3
                })
            
            if disk_warning:
                # This would typically involve execution of cleanup scripts on the node
                # For simulation, we'll just log this
                self.logger.info(f"Node {node_name} needs disk cleanup (disk usage: {row.get('disk_usage_percent', 0)}%)")
            
            if network_warning:
                # This would typically involve adjusting network policies
                # For simulation, we'll just log this
                self.logger.info(f"Node {node_name} has high network load ({row.get('network_load_mbps', 0)} Mbps)")
        
        return actions
    
    def decide_actions(self, node_metrics, pod_metrics, node_predictions, pod_predictions):
        """
        Decide all remediation actions based on current metrics and predictions
        
        Args:
            node_metrics: DataFrame with current node metrics
            pod_metrics: DataFrame with current pod metrics
            node_predictions: DataFrame with node predictions
            pod_predictions: DataFrame with pod predictions
        
        Returns:
            list: List of remediation actions to take, sorted by priority
        """
        actions = []
        
        # Get pod actions
        pod_actions = self.decide_pod_actions(pod_metrics, pod_predictions)
        actions.extend(pod_actions)
        
         # Get node actions
        node_actions = self.decide_node_actions(node_metrics, node_predictions)
        actions.extend(node_actions)
        
        # Sort actions by priority (lower number = higher priority)
        actions.sort(key=lambda x: x.get('priority', 99))
        
        return actions
    
    def execute_actions(self, actions):
        """
        Execute the decided remediation actions
        
        Args:
            actions: List of remediation actions to take
        
        Returns:
            list: Results of the executed actions
        """
        results = []
        
        for action in actions:
            action_type = action.get('action')
            params = action.get('params', {})
            reason = action.get('reason', '')
            
            self.logger.info(f"Executing action: {action_type} - Reason: {reason}")
            
            result = None
            
            # Execute the appropriate action based on action_type
            if action_type == 'restart_pod':
                result = self.actions.restart_pod(
                    namespace=params.get('namespace', 'default'),
                    pod_name=params.get('pod_name')
                )
            
            elif action_type == 'scale_deployment':
                # Handle relative scaling ('+' or '-')
                replicas = params.get('replicas')
                if isinstance(replicas, str) and (replicas.startswith('+') or replicas.startswith('-')):
                    # Get current replicas
                    try:
                        deployment = self.actions.apps_api.read_namespaced_deployment(
                            name=params.get('deployment_name'),
                            namespace=params.get('namespace', 'default')
                        )
                        current_replicas = deployment.spec.replicas
                        
                        if replicas.startswith('+'):
                            new_replicas = current_replicas + int(replicas[1:] or 1)
                        else:  # replicas.startswith('-')
                            new_replicas = max(1, current_replicas - int(replicas[1:] or 1))
                        
                        replicas = new_replicas
                    except Exception as e:
                        self.logger.error(f"Failed to get current replicas: {e}")
                        replicas = 1
                
                result = self.actions.scale_deployment(
                    namespace=params.get('namespace', 'default'),
                    deployment_name=params.get('deployment_name'),
                    replicas=int(replicas)
                )
            
            elif action_type == 'cordon_node':
                result = self.actions.cordon_node(
                    node_name=params.get('node_name')
                )
            
            elif action_type == 'drain_node':
                result = self.actions.drain_node(
                    node_name=params.get('node_name'),
                    delete_local_data=params.get('delete_local_data', False),
                    ignore_daemonsets=params.get('ignore_daemonsets', True)
                )
            
            elif action_type == 'evict_pod':
                result = self.actions.evict_pod(
                    namespace=params.get('namespace', 'default'),
                    pod_name=params.get('pod_name')
                )
            
            elif action_type == 'update_resource_limits':
                result = self.actions.update_resource_limits(
                    namespace=params.get('namespace', 'default'),
                    deployment_name=params.get('deployment_name'),
                    container_name=params.get('container_name'),
                    cpu_limit=params.get('cpu_limit'),
                    memory_limit=params.get('memory_limit')
                )
            
            elif action_type == 'add_toleration':
                result = self.actions.add_toleration(
                    namespace=params.get('namespace', 'default'),
                    deployment_name=params.get('deployment_name'),
                    key=params.get('key'),
                    operator=params.get('operator', 'Equal'),
                    value=params.get('value', ''),
                    effect=params.get('effect', 'NoSchedule')
                )
            
            elif action_type == 'taint_node':
                result = self.actions.taint_node(
                    node_name=params.get('node_name'),
                    key=params.get('key'),
                    value=params.get('value', ''),
                    effect=params.get('effect', 'NoSchedule')
                )
            
            else:
                self.logger.warning(f"Unknown action type: {action_type}")
                result = {
                    'success': False,
                    'action': action_type,
                    'message': f"Unknown action type: {action_type}"
                }
            
            # Add reason to result
            if result:
                result['reason'] = reason
            
            results.append(result)
        
        return results
    
    def _get_deployment_for_pod(self, pod_name):
        """
        Helper method to get the deployment name for a pod
        
        Args:
            pod_name: Pod name
        
        Returns:
            str: Deployment name or None if not found
        """
        # Extract deployment name from pod name (heuristic)
        # Most pods created by deployments have names like: deployment-name-random-suffix
        parts = pod_name.split('-')
        if len(parts) >= 2:
            # Try to find the deployment by removing the last part
            return '-'.join(parts[:-1])
        return None