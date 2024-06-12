def get_model(model_name, args):
    import ptvision.models.neonext as neonext
    return neonext.__dict__[model_name](num_classes=args.num_classes,
                                        drop_path=args.drop_path,
                                        conv_init=args.conv_init_type,
                                        shifts=args.shifts,
                                        layer_scale_init_value=args.layer_scale_init_value,
                                        linear_bias=args.linear_bias,
                                        kernel_spec=args.kernel_spec)
