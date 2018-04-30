from ignite.engines import Engine


def create_supervised_evaluator(model, inference_fn, metrics={}, cuda=False):
    """
    Factory function for creating an evaluator for supervised models.
    Extended version from ignite's create_supervised_evaluator
    Args:
        model (torch.nn.Module): the model to train
        inference_fn (function): inference function
        metrics (dict of str: Metric): a map of metric names to Metrics
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)
    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    engine = Engine(inference_fn)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine