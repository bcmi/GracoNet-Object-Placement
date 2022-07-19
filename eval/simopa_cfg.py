class Config(object):
    ## Path
    pretrained_model_path = None
    dataset_path = None
    img_path = None
    mask_path = None

    # * train
    train_data_path = None
    box_dic_path = None
    depth_feats_path = None
    train_reference_feature_path = None
    train_target_feature_path = None

    # * test
    test_data_path = None
    test_box_dic_path = None
    test_reference_feature_path = None
    test_target_feature_path = None

    ## Loader
    img_size = 256
    binary_mask_size = 64

    # * train
    num_workers = 4
    batch_size = 64
    base_lr = 1e-4
    lr_milestones = [10, 16]
    lr_gamma = 0.1
    epochs = 20
    eval_freq = 1
    save_freq = 5
    display_freq = 10

    ## Network
    class_num = 2
    geometric_feature_dim = 256
    roi_align_size = 3
    global_feature_size = 8
    attention_dim_head = 64

    # * reference head
    backbone = 'resnet18'
    relation_method = None
    attention_method = None
    refer_num = None
    attention_head = None
    without_mask = None
    without_global_feature = None

opt = Config()
