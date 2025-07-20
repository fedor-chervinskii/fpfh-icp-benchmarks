#!/usr/bin/env python3
"""Download BOP datasets for benchmarking."""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path


def download_file(url: str, output_path: str) -> None:
    """Download file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Local path to save file
    """
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f'\rDownloading... {percent}%')
        sys.stdout.flush()
    
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output_path, progress_hook)
    print("\nDownload completed!")


def extract_zip(zip_path: str, extract_path: str) -> None:
    """Extract ZIP file.
    
    Args:
        zip_path: Path to ZIP file
        extract_path: Directory to extract to
    """
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed!")


def download_dataset(dataset: str, output_path: str, subset: str = 'full') -> None:
    """Download a BOP dataset.
    
    Args:
        dataset: Dataset name
        output_path: Output directory
        subset: 'full' for complete dataset, 'bop' for BOP challenge subset
    """
    base_url = "https://bop.felk.cvut.cz/media/data/bop_datasets"
    
    # Dataset URLs from BOP HuggingFace Hub
    base_url = "https://huggingface.co/datasets/bop-benchmark"
    
    dataset_urls = {
        'lmo': {
            'base': f"{base_url}/lmo/resolve/main/lmo_base.zip",
            'models': f"{base_url}/lmo/resolve/main/lmo_models.zip",
            'test': f"{base_url}/lmo/resolve/main/lmo_test_all.zip",  # Full test set
            'test_bop': f"{base_url}/lmo/resolve/main/lmo_test_bop19.zip",  # BOP'19-24 subset
        },
        'ycbv': {
            'base': f"{base_url}/ycbv/resolve/main/ycbv_base.zip",
            'models': f"{base_url}/ycbv/resolve/main/ycbv_models.zip",
            'test': f"{base_url}/ycbv/resolve/main/ycbv_test_all.zip",
            'test_bop': f"{base_url}/ycbv/resolve/main/ycbv_test_bop19.zip",
        },
        'tless': {
            'base': f"{base_url}/tless/resolve/main/tless_base.zip",
            'models': f"{base_url}/tless/resolve/main/tless_models.zip",
            'test': f"{base_url}/tless/resolve/main/tless_test_primesense_all.zip",
            'test_bop': f"{base_url}/tless/resolve/main/tless_test_primesense_bop19.zip",
        },
        'itodd': {
            'base': f"{base_url}/itodd/resolve/main/itodd_base.zip",
            'models': f"{base_url}/itodd/resolve/main/itodd_models.zip",
            'test': f"{base_url}/itodd/resolve/main/itodd_test_all.zip",
            'test_bop': f"{base_url}/itodd/resolve/main/itodd_test_bop19.zip",
        },
        'hb': {
            'base': f"{base_url}/hb/resolve/main/hb_base.zip",
            'models': f"{base_url}/hb/resolve/main/hb_models.zip",
            'test': f"{base_url}/hb/resolve/main/hb_test_primesense_all.zip",
            'test_bop': f"{base_url}/hb/resolve/main/hb_test_primesense_bop19.zip",
        }
    }
    
    if dataset not in dataset_urls:
        print(f"Error: Unknown dataset '{dataset}'")
        print(f"Available datasets: {list(dataset_urls.keys())}")
        return
    
    # Create output directory
    dataset_dir = os.path.join(output_path, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download and extract each component
    for component, url in dataset_urls[dataset].items():
        # Skip test_bop if we want full dataset, or skip test if we want bop subset
        if subset == 'full' and component == 'test_bop':
            continue
        elif subset == 'bop' and component == 'test':
            continue
        
        zip_filename = f"{dataset}_{component}.zip"
        zip_path = os.path.join(dataset_dir, zip_filename)
        
        try:
            # Download
            download_file(url, zip_path)
            
            # Extract
            extract_zip(zip_path, dataset_dir)
            
            # Clean up ZIP file
            os.remove(zip_path)
            print(f"Removed {zip_filename}")
            
        except Exception as e:
            print(f"Error downloading {component}: {e}")
            continue
    
    print(f"\nDataset '{dataset}' downloaded to: {dataset_dir}")


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(
        description='Download BOP datasets for benchmarking'
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['lmo', 'ycbv', 'tless', 'itodd', 'hb', 'all'],
                       help='Dataset to download')
    parser.add_argument('--path', type=str, default='./data',
                       help='Output directory for datasets')
    parser.add_argument('--subset', type=str, default='full', 
                       choices=['full', 'bop'],
                       help='Download full dataset or BOP challenge subset only (default: full)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.path).mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'all':
        datasets = ['lmo', 'ycbv', 'tless', 'itodd', 'hb']
        for dataset in datasets:
            print(f"\n{'='*50}")
            print(f"Downloading {dataset.upper()} dataset ({'full' if args.subset == 'full' else 'BOP subset'})")
            print(f"{'='*50}")
            download_dataset(dataset, args.path, args.subset)
    else:
        print(f"Downloading {args.dataset.upper()} dataset ({'full' if args.subset == 'full' else 'BOP subset'})")
        download_dataset(args.dataset, args.path, args.subset)
    
    print("\n" + "="*50)
    print("SUCCESS!")
    print("="*50)
    print("Dataset downloaded successfully!")
    print("Visit https://bop.felk.cvut.cz/datasets/ for more information.")
    print("="*50)


if __name__ == '__main__':
    main()
