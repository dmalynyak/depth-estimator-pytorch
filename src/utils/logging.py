import os, csv, torch

class Logger():
    def __init__(self, model, save_log_path, save_checkpoint_path):
        self.log_path = save_log_path
        self.chkpt_path = save_checkpoint_path
        self.model = model

    def log_val_metrics_add(self, new_metrics, saved_metrics=None):

        if saved_metrics is None:
            return new_metrics
        
        for key in saved_metrics:
            if key == "parameters" or "saved":
                continue
            saved_metrics[key] += new_metrics[key]

        return saved_metrics


    def log_val_metrics_devide_batches(self, metrics, loader_len):

        assert metrics is not None, f"got empty metrics dictionary"

        for key in metrics:
            if key is "parameters" or "saved":
                continue
            metrics[key] = metrics[key] / loader_len

        return metrics

    def log_val_metrics_write(self, metrics):

        file_exists = os.path.exists(f"{self.log_path}/metrics.csv")

        metrics_only = {k: v for k, v in metrics.items() if k != "parameters"}
        first_cols = ["epoch", "saved"]
        other_cols = [k for k in metrics_only.keys() if k not in first_cols]
        ordered_fieldnames = first_cols + other_cols

        with open(f"{self.log_path}/metrics.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
            
            if not file_exists:
                if "parameters" in metrics:
                    f.write(f"# Parameters: {metrics['parameters']}\n")
                
                writer.writeheader()


            writer.writerow(metrics_only)

    def log_save_weights(self, metrics):
        torch.save(self.model.state_dict(), f"{self.chkpt_path}/best.pt")