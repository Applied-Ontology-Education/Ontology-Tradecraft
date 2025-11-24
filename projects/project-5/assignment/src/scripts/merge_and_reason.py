"""
scripts/merge_and_reason.py

Merge accepted_el.ttl with FacilityOntology.ttl using ROBOT,
reason with ELK, and save the final ontology as module_augmented.ttl
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def check_robot_installed():
    """Check if ROBOT is installed and accessible."""
    try:
        result = subprocess.run(
            ['robot', '--version'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info(f"✓ ROBOT found: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def download_robot():
    """Provide instructions for downloading ROBOT."""
    logger.error("ROBOT not found!")
    logger.error("\nTo install ROBOT:")
    logger.error("1. Download from: https://github.com/ontodev/robot/releases")
    logger.error("2. Extract robot.jar")
    logger.error("3. Add to PATH or use robot.jar directly")
    logger.error("\nWindows quick install:")
    logger.error("  1. Download robot.jar to your project directory")
    logger.error("  2. Use: java -jar robot.jar instead of 'robot' command")


def merge_and_reason(accepted_file: Path, base_ontology: Path, output_file: Path):
    """
    Merge ontologies and reason with ELK using ROBOT.
    
    Args:
        accepted_file: Path to accepted_el.ttl
        base_ontology: Path to FacilityOntology.ttl
        output_file: Path to output module_augmented.ttl
    """
    
    logger.info("="*60)
    logger.info("ONTOLOGY MERGE AND REASONING")
    logger.info("="*60)
    
    # Verify input files exist
    if not accepted_file.exists():
        raise FileNotFoundError(f"Accepted axioms file not found: {accepted_file}")
    
    if not base_ontology.exists():
        raise FileNotFoundError(f"Base ontology file not found: {base_ontology}")
    
    logger.info(f"Input 1: {accepted_file}")
    logger.info(f"Input 2: {base_ontology}")
    logger.info(f"Output: {output_file}")
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # ROBOT command to merge and reason
    # robot merge --input file1.ttl --input file2.ttl \
    #       reason --reasoner ELK \
    #       --output output.ttl
    
    logger.info("\nStep 1: Merging ontologies...")
    logger.info("Step 2: Reasoning with ELK...")
    logger.info("Step 3: Saving augmented ontology...")
    
    cmd = [
        'robot',
        'merge',
        '--input', str(accepted_file),
        '--input', str(base_ontology),
        'reason',
        '--reasoner', 'ELK',
        '--output', str(output_file)
    ]
    
    logger.info(f"\nExecuting: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            logger.info(f"ROBOT output: {result.stdout}")
        
        logger.info(f"\n✓ Successfully created augmented ontology: {output_file}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"\n✗ ROBOT command failed!")
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False
    
    except FileNotFoundError:
        logger.error("\n✗ ROBOT command not found!")
        logger.error("Make sure ROBOT is installed and in your PATH")
        logger.error("Or use: java -jar robot.jar instead of 'robot'")
        return False


def merge_and_reason_with_jar(accepted_file: Path, base_ontology: Path, 
                               output_file: Path, robot_jar: Path):
    """
    Merge ontologies and reason with ELK using robot.jar directly.
    
    Args:
        accepted_file: Path to accepted_el.ttl
        base_ontology: Path to FacilityOntology.ttl
        output_file: Path to output module_augmented.ttl
        robot_jar: Path to robot.jar file
    """
    
    logger.info("="*60)
    logger.info("ONTOLOGY MERGE AND REASONING (using robot.jar)")
    logger.info("="*60)
    
    # Verify input files exist
    if not accepted_file.exists():
        raise FileNotFoundError(f"Accepted axioms file not found: {accepted_file}")
    
    if not base_ontology.exists():
        raise FileNotFoundError(f"Base ontology file not found: {base_ontology}")
    
    if not robot_jar.exists():
        raise FileNotFoundError(f"robot.jar not found: {robot_jar}")
    
    logger.info(f"ROBOT JAR: {robot_jar}")
    logger.info(f"Input 1: {accepted_file}")
    logger.info(f"Input 2: {base_ontology}")
    logger.info(f"Output: {output_file}")
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nStep 1: Merging ontologies...")
    logger.info("Step 2: Reasoning with ELK...")
    logger.info("Step 3: Saving augmented ontology...")
    
    cmd = [
        'java', '-jar', str(robot_jar),
        'merge',
        '--input', str(accepted_file),
        '--input', str(base_ontology),
        'reason',
        '--reasoner', 'ELK',
        '--output', str(output_file)
    ]
    
    logger.info(f"\nExecuting: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            logger.info(f"ROBOT output: {result.stdout}")
        
        logger.info(f"\n✓ Successfully created augmented ontology: {output_file}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"\n✗ ROBOT command failed!")
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge ontologies with ROBOT and reason with ELK'
    )
    parser.add_argument(
        '--accepted',
        default='../generated/accepted_el.ttl',
        help='Path to accepted axioms file'
    )
    parser.add_argument(
        '--base',
        default='../FacilityOntology.ttl',
        help='Path to base ontology file'
    )
    parser.add_argument(
        '--output',
        default='../module_augmented.ttl',
        help='Path to output augmented ontology'
    )
    parser.add_argument(
        '--robot-jar',
        default=None,
        help='Path to robot.jar if not in PATH'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    
    # Handle accepted path
    if Path(args.accepted).is_absolute():
        accepted_path = Path(args.accepted).resolve()
    else:
        accepted_path = (script_dir / args.accepted).resolve()
    
    # Handle base path
    if Path(args.base).is_absolute():
        base_path = Path(args.base).resolve()
    else:
        base_path = (script_dir / args.base).resolve()
    
    # Handle output path
    if Path(args.output).is_absolute():
        output_path = Path(args.output).resolve()
    else:
        output_path = (script_dir / args.output).resolve()
    
    # Check if ROBOT is available
    if args.robot_jar:
        # Use robot.jar directly
        if Path(args.robot_jar).is_absolute():
            robot_jar_path = Path(args.robot_jar).resolve()
        else:
            robot_jar_path = (script_dir / args.robot_jar).resolve()
        
        if not robot_jar_path.exists():
            logger.error(f"robot.jar not found at: {robot_jar_path}")
            logger.error("\nDownload robot.jar from:")
            logger.error("https://github.com/ontodev/robot/releases/latest/download/robot.jar")
            sys.exit(1)
        
        success = merge_and_reason_with_jar(
            accepted_path,
            base_path,
            output_path,
            robot_jar_path
        )
    else:
        # Try to use ROBOT command
        if not check_robot_installed():
            download_robot()
            logger.error("\nAlternatively, download robot.jar and use:")
            logger.error("  python merge_and_reason.py --robot-jar path/to/robot.jar")
            sys.exit(1)
        
        success = merge_and_reason(
            accepted_path,
            base_path,
            output_path
        )
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("SUCCESS!")
        logger.info("="*60)
        logger.info(f"Augmented ontology saved to: {output_path}")
        logger.info("\nThe ontology includes:")
        logger.info("  - All axioms from FacilityOntology.ttl")
        logger.info("  - All accepted axioms from accepted_el.ttl")
        logger.info("  - Inferred axioms from ELK reasoner")
        logger.info("="*60)
        return 0
    else:
        logger.error("\n" + "="*60)
        logger.error("FAILED!")
        logger.error("="*60)
        logger.error("Check the error messages above")
        logger.error("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

    