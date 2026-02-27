"""
Automated Retraining Scheduler
Runs periodic checks and triggers retraining when needed
"""
import sys
import os
import time
import schedule
import logging
from datetime import datetime

sys.path.insert(0, os.getcwd() + '/src/pricemodel')

from retraining_pipeline import RetrainingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """Manages automated model retraining"""
    
    def __init__(self, 
                 performance_threshold=15.0,
                 time_threshold_days=30,
                 sliding_window_years=5):
        self.pipeline = RetrainingPipeline()
        self.performance_threshold = performance_threshold
        self.time_threshold_days = time_threshold_days
        self.sliding_window_years = sliding_window_years
        
        logger.info("Retraining scheduler initialized")
        logger.info(f"  Performance threshold: {performance_threshold}%")
        logger.info(f"  Time threshold: {time_threshold_days} days")
        logger.info(f"  Sliding window: {sliding_window_years} years")
    
    def check_and_retrain(self):
        """Check if retraining is needed and execute if necessary"""
        logger.info("="*80)
        logger.info("RETRAINING CHECK")
        logger.info("="*80)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        try:
            # Check if retraining needed
            should_retrain, reason = self.pipeline.should_retrain(
                performance_threshold=self.performance_threshold,
                time_threshold_days=self.time_threshold_days
            )
            
            if should_retrain:
                logger.info(f"Retraining triggered: {reason}")
                
                # Load and prepare data
                logger.info("Loading data...")
                df = self.pipeline.load_and_prepare_data(
                    sliding_window_years=self.sliding_window_years
                )
                
                logger.info(f"Data loaded: {len(df)} records")
                logger.info(f"Date range: {df['sale_date'].min()} to {df['sale_date'].max()}")
                
                # Train new model
                logger.info("Starting model training...")
                model = self.pipeline.train_new_model(df)
                
                logger.info("✓ Retraining completed successfully")
                logger.info(f"  Model saved to: {model.directory}")
                logger.info(f"  Mean Abs % Error: {model.dataframe['pct_error'].abs().mean():.2f}%")
                
                # Compare with previous models
                logger.info("\nModel comparison:")
                self.pipeline.compare_models(n_recent=5)
                
            else:
                logger.info(f"No retraining needed: {reason}")
                
                # Show active model info
                if self.pipeline.model_registry['active_model']:
                    active = self.pipeline.model_registry['active_model']
                    logger.info(f"Active model:")
                    logger.info(f"  Trained: {active['trained_date']}")
                    logger.info(f"  Error: {active['metrics']['mean_abs_pct_error']:.2f}%")
                    logger.info(f"  Records: {active['n_records']}")
        
        except Exception as e:
            logger.error(f"Retraining check failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info("="*80)
        logger.info("")
    
    def run_once(self):
        """Run a single check"""
        self.check_and_retrain()
    
    def run_scheduled(self, schedule_time="02:00"):
        """Run on a schedule"""
        logger.info(f"Scheduling daily retraining check at {schedule_time}")
        
        # Schedule daily check
        schedule.every().day.at(schedule_time).do(self.check_and_retrain)
        
        # Also run immediately on startup
        logger.info("Running initial check...")
        self.check_and_retrain()
        
        # Keep running
        logger.info("Scheduler running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
    
    def run_interval(self, hours=24):
        """Run at fixed intervals"""
        logger.info(f"Scheduling retraining check every {hours} hours")
        
        # Schedule interval check
        schedule.every(hours).hours.do(self.check_and_retrain)
        
        # Run immediately on startup
        logger.info("Running initial check...")
        self.check_and_retrain()
        
        # Keep running
        logger.info("Scheduler running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Model Retraining Scheduler')
    parser.add_argument('--mode', choices=['once', 'scheduled', 'interval'], 
                       default='scheduled',
                       help='Run mode: once, scheduled (daily), or interval')
    parser.add_argument('--time', default='02:00',
                       help='Time for scheduled mode (HH:MM format)')
    parser.add_argument('--interval', type=int, default=24,
                       help='Hours between checks for interval mode')
    parser.add_argument('--performance-threshold', type=float, default=15.0,
                       help='Performance threshold for retraining (%)')
    parser.add_argument('--time-threshold', type=int, default=30,
                       help='Time threshold for retraining (days)')
    parser.add_argument('--sliding-window', type=int, default=5,
                       help='Sliding window size (years)')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = RetrainingScheduler(
        performance_threshold=args.performance_threshold,
        time_threshold_days=args.time_threshold,
        sliding_window_years=args.sliding_window
    )
    
    # Run based on mode
    if args.mode == 'once':
        logger.info("Running single retraining check")
        scheduler.run_once()
    elif args.mode == 'scheduled':
        scheduler.run_scheduled(schedule_time=args.time)
    elif args.mode == 'interval':
        scheduler.run_interval(hours=args.interval)


if __name__ == "__main__":
    main()
