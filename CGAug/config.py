class Config:
    split = 'train'  # valid value: ['train', 'val']
    city_batch = 0  # valid value: range(4)
    ADE_root = '../../data'
    img_dir = '../../data/cityscapes/leftImg8bit/'
    mask_dir = '../../data/cityscapes/gtFine/'
    log_dir = '/inspurfs/group/hexm/ood_data/DTWP_ADE_seed_3'
    save_img_dir = '/inspurfs/group/hexm/ood_data/DTWP_ADE_seed_3/leftImg8bit/'
    save_mask_dir = '/inspurfs/group/hexm/ood_data/DTWP_ADE_seed_3/gtFine/'
    SAM_path = 'CGAug/pretrained_model/sam_vit_h_4b8939.pth'
    anomaly_weight_path = 'CGAug/pretrained_model/bt-f-xl.pth'
    controlnet_weight_path = 'CGAug/pretrained_model/control_sd15_seg.pth'

    WEATHER_LIST = ["cloudy", "rainy", "snowy", "foggy", "clear"]
    PLACE_PROMPT = [
        "New York City", "Tokyo", "The Bay Area", "London", "Singapore", "Los Angeles", "Hong Kong", "Beijing",
        "Shanghai", "Sydney", "Chicago", "Toronto", "Frankfurt", "Zurich", "Houston", "Seoul", "Melbourne", "Paris",
        "Geneva", "Dubai", "Mumbai", "Rome", "Seattle", "Shenzhen", "Osaka", "Boston", "Kyoto", "Miami", "Vancouver",
        "Tel Aviv", "Moscow", "Perth", "Brisbane", "Austin", "Hangzhou", "Delhi", "Madrid", "Auckland", "Abu Dhabi",
        "Manchester", "Nice", "Guangzhou", "Athens", "Doha", "Lisbon", "Dublin", "Riyadh", "Montreal", "Monaco",
        "Las Vegas", "Istanbul", "Warsaw", "Jerusalem", "San Diego", "Calgary", "Johannesburg", "Scottsdale",
        "Barcelona", "Milan", "Bengaluru", "Edinburgh", "Santa Barbara &amp; Montecito", "Kolkata",
        "Greenwich &amp; Darien", "Hyderabad", "West Palm Beach", "Ho Chi Minh City", "Florence", "Cairo", "Cape Town",
        "St. Petersburg", "Lagos", "Budapest", "Nairobi", "Netanya", "Herzliya", "Sharjah", "Durban", "Cape Winelands",
        "Garden Route", "Casablanca", "Pretoria", "Accra", "Luanda", "Dar Es Salaam", "Whale Coast", "Windhoek",
        "Marrakech", "Addis Ababa", "Kigali", "Maputo", "Mombasa", "Tangier", "Lusaka", "Swakopmund", "Walvis Bay",
    ]

    prompt = "An image sampled from various stereo video sequences taken by dash cam."
    a_prompt = 'best quality, extremely detailed, realistic, high resolution'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, time-lapse photography, blurry, pixelated, low resolution, cartoon, video progress bar，TV program icon，grey picture'

    save_memory = False
