import torch

def flatten_collate_fn(samples):
    """
    Collate function for a BloodSmearDataset dataloader. Images from each
    patient are gathered together in consecutive blocks and a key num_images
    allows to separate them for pooling.
    
    Args:
        samples: (list) list of items from the dataset chosen for the batch.
    
    Output:
        batch: (dict) dictionary containing the concatenated values of the instances
            for all bags in the batch. The keys are the same as in BloodSmearDataset,
            with the addition of num_images which specifies the number of instances
            in each bag contained in the batch.
    """
    batch_size = len(samples)
    num_images = [len(sample['images']) for sample in samples]
    batch = {
        'images': torch.zeros(sum(num_images), *samples[0]["images"][0].shape),
        'label': torch.zeros(batch_size),
        'patient_id': [],
        'gender': torch.zeros(batch_size),
        'age': torch.zeros(batch_size),
        'lymph_count': torch.zeros(batch_size),
        'num_images': torch.tensor(num_images),
    }
    idx_image = 0
    for idx_sample, sample in enumerate(samples):
        for image in sample['images']:
            batch['images'][idx_image] = image
            idx_image += 1
        batch['patient_id'].append(sample['patient_id'])
        for key in ['label', 'gender', 'age', 'lymph_count']:
            batch[key][idx_sample] = sample[key]
    return batch