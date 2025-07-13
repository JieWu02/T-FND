import torch


class WeiboConfig:

    # Paths
    data_path = './weibo_dataset/'
    output_path = './output/'
    
    # Image paths
    rumor_image_path = data_path + 'rumor_images/'
    nonrumor_image_path = data_path + 'nonrumor_images/'
    
    # Data file paths
    train_text_path = data_path + 'weibo_train.csv'
    validation_text_path = data_path + 'weibo_test.csv'
    test_text_path = data_path + 'weibo_test.csv'
    
    # Training parameters
    batch_size = 100
    epochs = 100
    num_workers = 1
    
    # Learning rates
    image_encoder_lr = 1.0e-05
    text_encoder_lr = 0.00016
    image_encoder_teacher_lr = 1.0e-05
    text_encoder_teacher_lr = 0.00016
    text_classifier_lr = 0.0003
    image_classifier_lr = 0.0003
    
    # Model architecture
    hidden_size = 128
    projection_size = 64
    dropout = 0.5
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Model names
    image_model_name = 'google/vit-base-patch16-224'
    text_encoder_model = "bert-base-chinese"
    text_tokenizer = "bert-base-chinese"
    
    # Model parameters
    image_embedding = 768
    text_embedding = 768
    max_length = 200
    size = 224  # image size
    
    # Training settings
    pretrained = True
    trainable = True
    
    # Classification
    classes = ['real', 'fake']
    
    # Loss weights
    # c_loss_weight = 1.0  # 未使用
    ce_loss_weight = 1.0
    ct_loss_weight = 1.0
    
    # Evaluation
    wanted_accuracy = 0.88
    
    # Early stopping
    patience = 5
    delta = 0.0000001
    
    # Two-stage training parameters
    first_stage_epochs = 5      # 第一阶段固定训练的epoch数
    second_stage_epochs = 50    # 第二阶段最大epoch数（可早停）
    
    # Gradient clipping
    max_grad_norm = 5.0
    
    # Weight decay for different components
    head_weight_decay = 0.001
    # attention_weight_decay = 0.001  # 未使用
    # classification_weight_decay = 0.0001  # 未使用
    
    # DatasetLoader reference
    def get_dataset_loader(self):
        from .data_loader import WeiboDatasetLoader
        return WeiboDatasetLoader
    class_weights = [1, 1]

