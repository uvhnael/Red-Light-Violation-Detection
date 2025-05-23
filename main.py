#!/usr/bin/env python3
# USAGE: python main.py --input videos/traffic.mp4 --output output/traffic_output.avi --model models/yolo12l.pt

import argparse
import logging
import os
from src.traffic_monitor import TrafficMonitor

def setup_logging(log_level=logging.INFO, logger_name='main'):
    """Setup root logger configuration"""
    # Check if root logger already has handlers to avoid duplicate configuration
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Get the named logger and set its level
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    return logger

def main():
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input video")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output video")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applying non-maxima suppression")
    ap.add_argument("-m", "--model", type=str, default="models/yolo12l.pt",
                    help="path to YOLO model file")
    ap.add_argument("-l", "--log-level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="logging level")
    ap.add_argument("-e", "--export", type=str, default="json",
                    choices=["json", "csv"],
                    help="export format for violation data")
    args = vars(ap.parse_args())
    
    # Setup logging
    log_level = getattr(logging, args["log_level"])
    logger = setup_logging(log_level)
    
    # Check if input file exists
    if not os.path.exists(args["input"]):
        logger.error(f"Input file does not exist: {args['input']}")
        return
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args["output"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create and run traffic monitor
    try:
        monitor = TrafficMonitor(
            input_path=args["input"],
            output_path=args["output"],
            model_path=args["model"],
            confidence=args["confidence"],
            threshold=args["threshold"],
            log_level=log_level,
            export_format=args["export"]
        )
        
        # Process the video
        monitor.process()
        
    except Exception as e:
        logger.error(f"Error in traffic monitoring: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main()
