from torchxrayvision.utils import normalize
import pydicom
import base64
import io
import torchvision.transforms as transforms
import torchxrayvision as xrv

def encode_file_to_base64(filepath):
    """
    Read a file and convert it to base64 string.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        str: Base64 encoded string of the file
    """
    with open(filepath, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")
    return encoded_string

def is_dicom_file(byte_data):
    # In a standard DICOM Part 10 file, 'DICM' should appear at bytes 128–131.
    return len(byte_data) > 132 and byte_data[128:132] == b'DICM'
    
def xrv_read_dicom_and_normalize_image(filepath_or_bytes):
    """
    Read a DICOM image file and return the normalized pixel array.
    
    Args:
        filepath (str): Path to the DICOM file
        
    Returns:
        numpy.ndarray: Normalized image array
    """
    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        filepath_or_bytes = io.BytesIO(filepath_or_bytes)
    
    img = pydicom.filereader.dcmread(filepath_or_bytes).pixel_array
    img = normalize(img, maxval=255, reshape=True)
    return img

def build_transforms():
    return  transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

