import nnsight
import os
from nnsight import CONFIG

from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
    print(nnsight.ndif_status())
