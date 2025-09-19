from configs.default_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # data
  data = config.data
  data.dataset = 'Dental_LD'
  #data.root = 'D:/우성_Denoising_Main_Vatech_Endo_fulldataset/De_noising_1st(high_dose)/De_noising_original/Data/Use_data'
  data.root = 'D:/우성_Denoising_Main_Vatech_Endo_fulldataset/De_noising_1st(high_dose)/De_noising_smooth_std_내컴/De_noising_smooth_Factor_21_내컴/Data/Use_data'
  data.test_in = 'PROJ_SKULL_PreMolarR_03_Slice'
  data.test_target = 'PROJ_SKULL_PreMolarR_345_Slice'
  data.is_complex = False
  data.is_multi = False
  
  data.image_size = 128

  data.resize_or_crop = False
  data.crop_size = 128
  data.normal_Option = False
  data.remove_back_patch = False
               
  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.ation = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config