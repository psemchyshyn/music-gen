    # trainer = Trainer(progress_bar_refresh_rate=100,
    #                 max_epochs=12, 
    #                 check_val_every_n_epoch=1,
    #                 callbacks=[estopping_callback, checkpoint_callback],
    #                 auto_lr_find=True,
    #                 track_grad_norm=2,
    #                 num_sanity_val_steps=1,
    #                 default_root_dir="logs",
    #                 weights_save_path="checkpoints"
    #                 )