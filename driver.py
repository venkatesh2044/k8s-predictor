from remediation.action_library import RemediationActions
from remediation.decision_engine import RemediationDecisionEngine
from remediation.kubernetes_api import KubernetesClient

class KubernetesPredictorDriver:
    """
    Main driver for the Kubernetes Crash Prediction and Remediation System
    """
    def __init__(self, config_file=None):
        """
        Initialize the driver
        
        Args:
            config_file: Path to the configuration file
        """
        # Load configuration
        self.config = Config(config_file)
        
        # Set up logging
        self.logger = setup_logging(
            level=self.config.get('logging.level', 'INFO'),
            log_file=self.config.get('logging.file'),
            log_to_stdout=True,
            name='k8s-predictor'
        )
        
        self.logger.info("Initializing Kubernetes Predictor Driver")
        
        # Initialize components
        self._init_components()
        
        # Metrics storage
        self.node_metrics_history = []
        self.pod_metrics_history = []
        
        # Statistics
        self.stats = {
            'cycles': 0,
            'predictions': 0,
            'anomalies_detected': 0,
            'remediation_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0
        }
    
    def _init_components(self):
        """Initialize all components"""
        try:
            # Data collection and feature engineering
            self.data_collector = DataCollector(
                prometheus_url=self.config.get('monitoring.prometheus_url'),
                use_kube_config=self.config.get('monitoring.use_kube_config', False),
                in_cluster=self.config.get('monitoring.in_cluster', True)
            )
            
            self.feature_engineer = FeatureEngineer()
            
            # Prediction models
            model_dir = self.config.get('prediction.model_dir', './models')
            os.makedirs(model_dir, exist_ok=True)
            
            self.resource_predictor = ResourcePredictor(model_dir=model_dir)
            self.pod_crash_predictor = PodCrashPredictor(model_dir=model_dir)
            self.node_failure_predictor = NodeFailurePredictor(model_dir=model_dir)
            self.network_issue_predictor = NetworkIssuePredictor(model_dir=model_dir)
            
            # Remediation components
            self.kubernetes_client = KubernetesClient(
                use_kube_config=self.config.get('monitoring.use_kube_config', False),
                in_cluster=self.config.get('monitoring.in_cluster', True)
            )
            
            self.remediation_actions = RemediationActions(
                use_kube_config=self.config.get('monitoring.use_kube_config', False),
                in_cluster=self.config.get('monitoring.in_cluster', True)
            )
            
            self.decision_engine = RemediationDecisionEngine(
                action_library=self.remediation_actions
            )
            
            # Visualization
            self.visualizer = Visualizer(
                output_dir=self.config.get('visualization.output_dir', './visualizations')
            )
            
            self.logger.info("Successfully initialized all components")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_training_cycle(self, node_metrics=None, pod_metrics=None):
        """
        Run a training cycle
        
        Args:
            node_metrics: Node metrics DataFrame (if None, collect from cluster)
            pod_metrics: Pod metrics DataFrame (if None, collect from cluster)
        """
        self.logger.info("Running training cycle")
        
        try:
            # Collect metrics if not provided
            if node_metrics is None or pod_metrics is None:
                node_metrics, pod_metrics = self.data_collector.collect_all_metrics()
            
            # Prepare features
            node_features = self.feature_engineer.prepare_node_features(node_metrics, fit_scaler=True)
            pod_features = self.feature_engineer.prepare_pod_features(pod_metrics, fit_scaler=True)
            
            if node_features.empty or pod_features.empty:
                self.logger.warning("No data available for training")
                return
            
            # Train resource predictors
            self.logger.info("Training resource prediction models")
            
            # Node resource predictors
            self.resource_predictor.train_node_models(
                X=node_features.drop(['node_name', 'timestamp', 'node_condition'], axis=1, errors='ignore'),
                y_cpu=node_metrics['cpu_usage_percent'],
                y_memory=node_metrics['memory_usage_percent'],
                y_disk=node_metrics['disk_usage_percent'],
                y_network=node_metrics['network_load_mbps']
            )
            
            # Pod resource predictors
            self.resource_predictor.train_pod_models(
                X=pod_features.drop(['pod_name', 'pod_namespace', 'node_name', 'timestamp', 'pod_status', 'container_status'], axis=1, errors='ignore'),
                y_cpu=pod_metrics['pod_cpu_usage_percent'],
                y_memory=pod_metrics['pod_memory_usage_percent'],
                y_network=pod_metrics['pod_network_mbps']
            )
            
            # Train crash predictors
            self.logger.info("Training crash prediction models")
            
            # Create target variables
            pod_crash_target = (pod_metrics['issue_label'].str.contains('crash') | 
                               (pod_metrics['restart_count'] > 5)).astype(int)
            
            node_failure_target = (node_metrics['node_condition'] != 'True').astype(int)
            
            network_issue_target = (node_metrics['network_errors'] > 5).astype(int)
            
            # Train pod crash predictor
            self.pod_crash_predictor.train(
                X=pod_features.drop(['pod_name', 'pod_namespace', 'node_name', 'timestamp', 'pod_status', 'container_status'], axis=1, errors='ignore'),
                y=pod_crash_target
            )
            
            # Train node failure predictor
            self.node_failure_predictor.train(
                X=node_features.drop(['node_name', 'timestamp', 'node_condition'], axis=1, errors='ignore'),
                y=node_failure_target
            )
            
            # Train network issue predictor
            self.network_issue_predictor.train(
                X=node_features.drop(['node_name', 'timestamp', 'node_condition'], axis=1, errors='ignore'),
                y=network_issue_target
            )
            
            self.logger.info("Successfully completed training cycle")
            
        except Exception as e:
            self.logger.error(f"Error during training cycle: {e}")
    
    def run_prediction_cycle(self):
        """
        Run a prediction cycle
        
        Returns:
            tuple: Node predictions, pod predictions
        """
        self.logger.info("Running prediction cycle")
        
        try:
            # Collect metrics
            node_metrics, pod_metrics = self.data_collector.collect_all_metrics()
            
            # Store metrics in history
            self.node_metrics_history.append(node_metrics)
            self.pod_metrics_history.append(pod_metrics)
            
            # Trim history to keep only recent data
            max_history = 100
            if len(self.node_metrics_history) > max_history:
                self.node_metrics_history = self.node_metrics_history[-max_history:]
            if len(self.pod_metrics_history) > max_history:
                self.pod_metrics_history = self.pod_metrics_history[-max_history:]
            
            # Prepare features
            node_features = self.feature_engineer.prepare_node_features(node_metrics)
            pod_features = self.feature_engineer.prepare_pod_features(pod_metrics)
            
            if node_features.empty or pod_features.empty:
                self.logger.warning("No data available for prediction")
                return None, None
            
            # Make predictions
            node_resource_pred = self.resource_predictor.predict_node_resources(
                X=node_features.drop(['node_name', 'timestamp', 'node_condition'], axis=1, errors='ignore')
            )
            
            pod_resource_pred = self.resource_predictor.predict_pod_resources(
                X=pod_features.drop(['pod_name', 'pod_namespace', 'node_name', 'timestamp', 'pod_status', 'container_status'], axis=1, errors='ignore')
            )
            
            pod_crash_pred = self.pod_crash_predictor.predict(
                X=pod_features.drop(['pod_name', 'pod_namespace', 'node_name', 'timestamp', 'pod_status', 'container_status'], axis=1, errors='ignore')
            )
            
            node_failure_pred = self.node_failure_predictor.predict(
                X=node_features.drop(['node_name', 'timestamp', 'node_condition'], axis=1, errors='ignore')
            )
            
            network_issue_pred = self.network_issue_predictor.predict(
                X=node_features.drop(['node_name', 'timestamp', 'node_condition'], axis=1, errors='ignore')
            )
            
            # Combine predictions
            node_predictions = pd.concat([
                node_metrics[['node_name']],
                node_resource_pred,
                node_failure_pred,
                network_issue_pred
            ], axis=1)
            
            pod_predictions = pd.concat([
                pod_metrics[['pod_name', 'pod_namespace', 'node_name']],
                pod_resource_pred,
                pod_crash_pred
            ], axis=1)
            
            # Count predictions
            self.stats['predictions'] += len(node_predictions) + len(pod_predictions)
            
            # Count anomalies
            anomalies = 0
            anomalies += node_predictions['cpu_warning'].sum()
            anomalies += node_predictions['memory_warning'].sum()
            anomalies += node_predictions['disk_warning'].sum()
            anomalies += node_predictions['network_warning'].sum()
            anomalies += node_predictions['failure_predicted'].sum()
            anomalies += node_predictions['network_issue_predicted'].sum()
            anomalies += pod_predictions['cpu_warning'].sum()
            anomalies += pod_predictions['memory_warning'].sum()
            anomalies += pod_predictions['crash_predicted'].sum()
            
            self.stats['anomalies_detected'] += anomalies
            
            self.logger.info(f"Prediction cycle completed. Detected {anomalies} potential issues.")
            
            return node_metrics, node_predictions, pod_metrics, pod_predictions
            
        except Exception as e:
            self.logger.error(f"Error during prediction cycle: {e}")
            return None, None, None, None
    
    def run_remediation_cycle(self, node_metrics, node_predictions, pod_metrics, pod_predictions):
        """
        Run a remediation cycle
        
        Args:
            node_metrics: Node metrics DataFrame
            node_predictions: Node predictions DataFrame
            pod_metrics: Pod metrics DataFrame
            pod_predictions: Pod predictions DataFrame
        
        Returns:
            list: Remediation results
        """
        self.logger.info("Running remediation cycle")
        
        if not self.config.get('remediation.enabled', True):
            self.logger.info("Remediation is disabled in configuration")
            return []
        
        try:
            # Decide actions
            actions = self.decision_engine.decide_actions(
                node_metrics=node_metrics,
                pod_metrics=pod_metrics,
                node_predictions=node_predictions,
                pod_predictions=pod_predictions
            )
            
            # Limit number of actions per cycle
            max_actions = self.config.get('remediation.max_actions_per_cycle', 5)
            if len(actions) > max_actions:
                self.logger.warning(f"Limiting remediation actions to {max_actions} (out of {len(actions)})")
                actions = actions[:max_actions]
            
            # Execute actions
            if self.config.get('remediation.approval_required', False):
                self.logger.info(f"Remediation actions require approval. Proposed {len(actions)} actions.")
                return actions
            else:
                results = self.decision_engine.execute_actions(actions)
                
                # Update statistics
                self.stats['remediation_actions'] += len(actions)
                self.stats['successful_actions'] += sum(1 for r in results if r.get('success', False))
                self.stats['failed_actions'] += sum(1 for r in results if not r.get('success', False))
                
                self.logger.info(f"Executed {len(results)} remediation actions. Success: {sum(1 for r in results if r.get('success', False))}, Failed: {sum(1 for r in results if not r.get('success', False))}")
                
                return results
            
        except Exception as e:
            self.logger.error(f"Error during remediation cycle: {e}")
            return []
    
    def run_cycle(self):
        """
        Run a complete monitoring, prediction, and remediation cycle
        
        Returns:
            tuple: Results of the cycle
        """
        self.stats['cycles'] += 1
        cycle_start = time.time()
        
        # Run prediction cycle
        node_metrics, node_predictions, pod_metrics, pod_predictions = self.run_prediction_cycle()
        
        if node_metrics is None or pod_metrics is None:
            self.logger.warning("Skipping cycle due to missing metrics")
            return None, None, None
        
        # Run remediation cycle
        remediation_results = self.run_remediation_cycle(
            node_metrics=node_metrics,
            node_predictions=node_predictions,
            pod_metrics=pod_metrics,
            pod_predictions=pod_predictions
        )
        
        cycle_duration = time.time() - cycle_start
        self.logger.info(f"Completed cycle in {cycle_duration:.2f} seconds")
        
        return node_predictions, pod_predictions, remediation_results
    
    def run(self, cycles=None, interval=None):
        """
        Run the driver continuously
        
        Args:
            cycles: Number of cycles to run (None for infinite)
            interval: Interval between cycles in seconds (None to use config value)
        """
        if interval is None:
            interval = self.config.get('monitoring.interval', 60)
        
        self.logger.info(f"Starting Kubernetes Predictor Driver (interval: {interval}s)")
        
        # Run initial training cycle
        try:
            self.logger.info("Running initial training")
            self.run_training_cycle()
        except Exception as e:
            self.logger.error(f"Error during initial training: {e}")
        
        # Main loop
        cycle_count = 0
        try:
            while cycles is None or cycle_count < cycles:
                cycle_count += 1
                self.logger.info(f"Starting cycle {cycle_count}")
                
                # Run a cycle
                self.run_cycle()
                
                # Sleep until next cycle
                time.sleep(interval)
        except KeyboardInterrupt:
            self.logger.info("Stopping driver due to keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.logger.info(f"Driver stopped after {cycle_count} cycles")
            self.print_stats()
    
    def print_stats(self):
        """Print driver statistics"""
        self.logger.info("Driver Statistics:")
        self.logger.info(f"  Total cycles: {self.stats['cycles']}")
        self.logger.info(f"  Total predictions: {self.stats['predictions']}")
        self.logger.info(f"  Anomalies detected: {self.stats['anomalies_detected']}")
        self.logger.info(f"  Remediation actions: {self.stats['remediation_actions']}")
        self.logger.info(f"  Successful actions: {self.stats['successful_actions']}")
        self.logger.info(f"  Failed actions: {self.stats['failed_actions']}")
        
        if self.stats['remediation_actions'] > 0:
            success_rate = self.stats['successful_actions'] / self.stats['remediation_actions'] * 100
            self.logger.info(f"  Remediation success rate: {success_rate:.2f}%")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Kubernetes Crash Prediction and Remediation System')
    parser.add_argument('-c', '--config', help='Path to configuration file')
    parser.add_argument('-t', '--train', action='store_true', help='Run training cycle only')
    parser.add_argument('-p', '--predict', action='store_true', help='Run prediction cycle only')
    parser.add_argument('-n', '--cycles', type=int, help='Number of cycles to run')
    parser.add_argument('-i', '--interval', type=int, help='Interval between cycles in seconds')
    parser.add_argument('-g', '--generate-data', action='store_true', help='Generate sample dataset and exit')
    parser.add_argument('-o', '--output', help='Output file for generated dataset')
    parser.add_argument('-s', '--size', type=int, default=20000, help='Size of generated dataset')
    return parser.parse_args()

def generate_sample_dataset(size=20000, output_file=None):
    """
    Generate a sample dataset for training and testing
    
    Args:
        size: Number of records to generate
        output_file: Output file path
    
    Returns:
        tuple: Node metrics DataFrame, pod metrics DataFrame
    """
    import random
    from datetime import datetime, timedelta
    
    # Set up logging
    logger = setup_logging(level='INFO', log_to_stdout=True, name='data-generator')
    logger.info(f"Generating sample dataset with {size} records")
    
    # Generate timestamps (last 5 days)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=5)
    timestamps = [start_time + timedelta(seconds=i * (5*24*60*60 / size)) for i in range(size)]
    
    # Generate node names
    node_count = 10
    nodes = [f'node-{i}' for i in range(node_count)]
    
    # Generate pod names (40 pods per node)
    pod_count = node_count * 40
    pods = []
    for i in range(pod_count):
        node_idx = i % node_count
        pods.append({
            'pod_name': f'pod-{i}',
            'pod_namespace': random.choice(['default', 'kube-system', 'application', 'monitoring']),
            'node_name': f'node-{node_idx}'
        })
    
    # Generate node metrics
    node_metrics = []
    for ts in timestamps:
        for node in nodes:
            # Base values
            cpu_base = random.uniform(30, 70)
            memory_base = random.uniform(40, 60)
            disk_base = random.uniform(50, 70)
            network_base = random.uniform(20, 40)
            
            # Add some randomness
            cpu = min(100, max(0, cpu_base + random.uniform(-5, 5)))
            memory = min(100, max(0, memory_base + random.uniform(-5, 5)))
            disk = min(100, max(0, disk_base + random.uniform(-2, 2)))
            network = max(0, network_base + random.uniform(-5, 5))
            
            # Add some spikes/anomalies (5% chance)
            if random.random() < 0.05:
                # Choose which metric to spike
                spike_type = random.choice(['cpu', 'memory', 'disk', 'network'])
                if spike_type == 'cpu':
                    cpu = min(100, cpu * random.uniform(1.3, 1.8))
                elif spike_type == 'memory':
                    memory = min(100, memory * random.uniform(1.3, 1.8))
                elif spike_type == 'disk':
                    disk = min(100, disk * random.uniform(1.1, 1.3))
                elif spike_type == 'network':
                    network = network * random.uniform(1.5, 3.0)
            
            # Generate other metrics
            pod_count = random.randint(10, 40)
            api_latency = random.uniform(1, 10)
            etcd_latency = random.uniform(1, 5)
            network_errors = max(0, int(network / 20))
            network_drops = max(0, int(network / 30))
            response_time = max(1, api_latency + etcd_latency + random.uniform(0, 5))
            
            # Generate node condition (1% chance of not being Ready)
            node_condition = 'True' if random.random() > 0.01 else 'False'
            
            # Calculate memory in bytes
            total_memory = random.randint(8, 64) * 1024 * 1024 * 1024  # 8-64 GB
            available_memory = int(total_memory * (1 - memory / 100))
            
            # Calculate disk in bytes
            total_disk = random.randint(100, 1000) * 1024 * 1024 * 1024  # 100-1000 GB
            available_disk = int(total_disk * (1 - disk / 100))
            
            # Create node metric record
            node_metrics.append({
                'timestamp': ts.timestamp(),
                'node_name': node,
                'cpu_usage_percent': cpu,
                'memory_usage_percent': memory,
                'disk_usage_percent': disk,
                'network_load_mbps': network,
                'pod_count': pod_count,
                'api_server_latency_ms': api_latency,
                'etcd_latency_ms': etcd_latency,
                'node_condition': node_condition,
                'available_memory_bytes': available_memory,
                'total_memory_bytes': total_memory,
                'available_disk_bytes': available_disk,
                'total_disk_bytes': total_disk,
                'network_errors': network_errors,
                'network_drops': network_drops,
                'response_time_ms': response_time
            })
    
    # Generate pod metrics
    pod_metrics = []
    for ts in timestamps:
        # Only include a subset of pods for each timestamp (to keep dataset size reasonable)
        sample_pods = random.sample(pods, min(len(pods), size // 10))
        
        for pod in sample_pods:
            # Base values with some randomness
            cpu = random.uniform(10, 50)
            memory = random.uniform(20, 60)
            network = random.uniform(1, 20)
            
            # Get node's current state (approximate to closest node metric)
            node_name = pod['node_name']
            node_state = next((n for n in node_metrics if n['node_name'] == node_name and 
                             abs(n['timestamp'] - ts.timestamp()) < 60), None)
            
            # Adjust pod metrics based on node state (if available)
            if node_state:
                # If node is under high resource usage, pods are likely affected
                if node_state['cpu_usage_percent'] > 80:
                    cpu = min(100, cpu * random.uniform(1.2, 1.5))
                if node_state['memory_usage_percent'] > 80:
                    memory = min(100, memory * random.uniform(1.2, 1.5))
                if node_state['network_load_mbps'] > 60:
                    network = network * random.uniform(1.2, 1.5)
            
            # Restart count (most pods have 0, some have a few, few have many)
            restart_distribution = random.random()
            if restart_distribution < 0.8:  # 80% of pods
                restart_count = 0
            elif restart_distribution < 0.95:  # 15% of pods
                restart_count = random.randint(1, 3)
            else:  # 5% of pods
                restart_count = random.randint(4, 20)
            
            # Pod status (most are Running, some are Failed or other)
            status_distribution = random.random()
            if status_distribution < 0.95:  # 95% of pods
                pod_status = 'Running'
                container_status = 'running'
            elif status_distribution < 0.98:  # 3% of pods
                pod_status = 'Failed'
                container_status = 'terminated'
            elif status_distribution < 0.99:  # 1% of pods
                pod_status = 'Pending'
                container_status = 'waiting'
            else:  # 1% of pods
                pod_status = 'Unknown'
                container_status = 'unknown'
            
            # Set issue label based on metrics and status
            issue_label = 'none'
            if pod_status == 'Failed':
                issue_label = 'pod_crash'
            elif restart_count > 5:
                issue_label = 'frequent_restarts'
            elif cpu > 85:
                issue_label = 'resource_exhaustion_cpu'
            elif memory > 85:
                issue_label = 'resource_exhaustion_memory'
            
            # Create pod metric record
            pod_metrics.append({
                'timestamp': ts.timestamp(),
                'pod_name': pod['pod_name'],
                'pod_namespace': pod['pod_namespace'],
                'node_name': pod['node_name'],
                'pod_status': pod_status,
                'container_status': container_status,
                'pod_cpu_usage_percent': cpu,
                'pod_memory_usage_percent': memory,
                'pod_network_mbps': network,
                'restart_count': restart_count,
                'issue_label': issue_label
            })
    
    # Convert to DataFrames
    node_df = pd.DataFrame(node_metrics)
    pod_df = pd.DataFrame(pod_metrics)
    
    logger.info(f"Generated {len(node_df)} node metrics and {len(pod_df)} pod metrics")
    
    # Save to file if specified
    if output_file:
        # Save as CSV
        if output_file.endswith('.csv'):
            # Save separate files for nodes and pods
            node_file = output_file.replace('.csv', '_nodes.csv')
            pod_file = output_file.replace('.csv', '_pods.csv')
            
            node_df.to_csv(node_file, index=False)
            pod_df.to_csv(pod_file, index=False)
            
            logger.info(f"Saved node metrics to {node_file}")
            logger.info(f"Saved pod metrics to {pod_file}")
        # Save as single CSV with both datasets
        else:
            # Add type column to distinguish between node and pod metrics
            node_df['metric_type'] = 'node'
            pod_df['metric_type'] = 'pod'
            
            # Combine datasets
            combined_df = pd.concat([node_df, pod_df], ignore_index=True)
            
            # Save to file
            combined_file = output_file if output_file.endswith('.csv') else output_file + '.csv'
            combined_df.to_csv(combined_file, index=False)
            
            logger.info(f"Saved combined metrics to {combined_file}")
    
    return node_df, pod_df

def main():
    """Main function"""
    args = parse_args()
    
    # Generate sample dataset if requested
    if args.generate_data:
        generate_sample_dataset(size=args.size, output_file=args.output or 'kubernetes_metrics_dataset.csv')
        return
    
    # Create driver
    driver = KubernetesPredictorDriver(config_file=args.config)
    
    # Run specific cycle if requested
    if args.train:
        driver.run_training_cycle()
    elif args.predict:
        node_metrics, node_predictions, pod_metrics, pod_predictions = driver.run_prediction_cycle()
        if node_predictions is not None and pod_predictions is not None:
            print("Node Predictions:")
            print(node_predictions.head())
            print("\nPod Predictions:")
            print(pod_predictions.head())
    else:
        # Run continuously
        driver.run(cycles=args.cycles, interval=args.interval)

if __name__ == "__main__":
    main()