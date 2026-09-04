import os, csv

class Logger():
    def __init__(self, save_file_path):
        self.path = save_file_path

    def log_val_metrics_add(self, new_metrics, saved_metrics=None):

        if saved_metrics is None:
            return new_metrics
        
        for key in saved_metrics:
            if key == "parameters":
                continue
            saved_metrics[key] += new_metrics[key]

        return saved_metrics


    def log_val_metrics_devide_batches(self, metrics, loader_len):

        assert metrics is not None, f"got empty metrics dictionary"

        for key in metrics:
            if key is "parameters":
                continue
            metrics[key] = metrics[key] / loader_len

        return metrics

    def log_val_metrics_write(self, metrics):

        file_exists = os.path.exists(self.path)

        metrics_only = {k: v for k, v in metrics.items() if k != "parameters"}
        
        with open(self.path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_only.keys())
            
            if not file_exists:
                if "parameters" in metrics:
                    f.write(f"Parameters: {metrics['parameters']}\n")
                
                writer.writeheader()

            writer.writerow(metrics_only)