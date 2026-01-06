#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knight Online Upgrade Data Visualization

Creates comprehensive visualizations of upgrade statistics for video presentation
"""

import os
import re
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime


class UpgradeVisualizer:
    def __init__(self, game_states_folder="game_states"):
        """Initialize the upgrade visualizer"""
        self.game_states_folder = game_states_folder
        
        # Storage for aggregated data
        self.total_upgrade_data = {
            "3_to_4": {"attempts": 0, "successes": 0, "failures": 0},
            "4_to_5": {"attempts": 0, "successes": 0, "failures": 0},
            "5_to_6": {"attempts": 0, "successes": 0, "failures": 0},
            "6_to_7": {"attempts": 0, "successes": 0, "failures": 0}
        }
        
        self.total_coins_spent = 0
        self.total_upgrades = 0
        self.total_plus_seven_achieved = 0
        self.sessions_count = 0
        
        # Set dark theme for cool gaming aesthetic
        plt.style.use('dark_background')
        
    def find_upgrade_files(self):
        """Find all upgrade statistics files"""
        pattern = os.path.join(self.game_states_folder, "upgrade_stats_*.txt")
        files = glob.glob(pattern)
        files.sort()
        return files
    
    def parse_upgrade_file(self, filepath):
        """Parse a single upgrade statistics file"""
        session_data = {
            "upgrade_data": {
                "3_to_4": {"attempts": 0, "successes": 0, "failures": 0},
                "4_to_5": {"attempts": 0, "successes": 0, "failures": 0},
                "5_to_6": {"attempts": 0, "successes": 0, "failures": 0},
                "6_to_7": {"attempts": 0, "successes": 0, "failures": 0}
            },
            "total_upgrades": 0,
            "total_cost": 0,
            "plus_seven_achieved": 0
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Extract totals
                upgrades_match = re.search(r'Total Upgrades: (\d+)', content)
                if upgrades_match:
                    session_data["total_upgrades"] = int(upgrades_match.group(1))
                
                cost_match = re.search(r'Total Cost: ([\d,]+) coins', content)
                if cost_match:
                    cost_str = cost_match.group(1).replace(',', '')
                    session_data["total_cost"] = int(cost_str)
                
                plus_seven_match = re.search(r'\+7 Items Achieved: (\d+)', content)
                if plus_seven_match:
                    session_data["plus_seven_achieved"] = int(plus_seven_match.group(1))
                
                # Extract upgrade statistics
                upgrade_patterns = {
                    "3_to_4": r'\+3 → \+to → \+4 Upgrades:\s*Attempts: (\d+)\s*Successes: (\d+)\s*Failures: (\d+)',
                    "4_to_5": r'\+4 → \+to → \+5 Upgrades:\s*Attempts: (\d+)\s*Successes: (\d+)\s*Failures: (\d+)',
                    "5_to_6": r'\+5 → \+to → \+6 Upgrades:\s*Attempts: (\d+)\s*Successes: (\d+)\s*Failures: (\d+)',
                    "6_to_7": r'\+6 → \+to → \+7 Upgrades:\s*Attempts: (\d+)\s*Successes: (\d+)\s*Failures: (\d+)'
                }
                
                for level, pattern in upgrade_patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        session_data["upgrade_data"][level] = {
                            "attempts": int(match.group(1)),
                            "successes": int(match.group(2)),
                            "failures": int(match.group(3))
                        }
                
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
        
        return session_data
    
    def load_all_data(self):
        """Load and aggregate all upgrade data"""
        files = self.find_upgrade_files()
        
        if not files:
            print("❌ No upgrade statistics files found!")
            return False
        
        print(f"📊 Loading data from {len(files)} files...")
        
        for filepath in files:
            session_data = self.parse_upgrade_file(filepath)
            
            if session_data:
                self.sessions_count += 1
                self.total_upgrades += session_data["total_upgrades"]
                self.total_coins_spent += session_data["total_cost"]
                self.total_plus_seven_achieved += session_data["plus_seven_achieved"]
                
                # Aggregate upgrade statistics
                for level in self.total_upgrade_data:
                    self.total_upgrade_data[level]["attempts"] += session_data["upgrade_data"][level]["attempts"]
                    self.total_upgrade_data[level]["successes"] += session_data["upgrade_data"][level]["successes"]
                    self.total_upgrade_data[level]["failures"] += session_data["upgrade_data"][level]["failures"]
        
        print(f"✅ Loaded {self.sessions_count} sessions")
        return True
    
    def create_visualization(self):
        """Create comprehensive visualization"""
        # Create figure with cleaner layout
        fig = plt.figure(figsize=(20, 11))
        fig.patch.set_facecolor('#0a0819')
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3, 
                     height_ratios=[2.5, 1], width_ratios=[1, 1])
        
        # Color scheme - Knight Online themed
        colors = {
            'success': '#00ff88',
            'failure': '#ff4444',
            'neutral': '#8b2be2',
            'gold': '#ffd700',
            'blue': '#4169e1'
        }
        
        # 1. Success vs Failure Bar Chart (Top - spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_success_failure_bars(ax1, colors)
        
        # 2. Success Rates Info (Bottom Left)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_success_rates_info(ax2, colors)
        
        # 3. Cost Analysis (Bottom Right)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_cost_analysis(ax3, colors)
        
        # Main title
        fig.suptitle('KNIGHT ONLINE UPGRADE ANALYSIS', 
                     fontsize=28, fontweight='bold', color=colors['gold'], y=0.97)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upgrade_visualization_{timestamp}.png"
        plt.savefig(filename, dpi=300, facecolor='#0a0819', edgecolor='none', bbox_inches='tight')
        print(f"💾 Visualization saved: {filename}")
        
        return filename
    
    def _plot_cost_analysis(self, ax, colors):
        """Plot cost analysis"""
        ax.axis('off')
        
        if self.total_plus_seven_achieved > 0:
            cost_per_plus_seven = self.total_coins_spent / self.total_plus_seven_achieved
        else:
            cost_per_plus_seven = 0
        
        if self.total_upgrades > 0:
            cost_per_upgrade = self.total_coins_spent / self.total_upgrades
        else:
            cost_per_upgrade = 0
        
        stats_text = f"""
COST ANALYSIS

Total Spent: {self.total_coins_spent:,.0f}

+7 Items: {self.total_plus_seven_achieved}

Cost per +7: {cost_per_plus_seven:,.0f}

Cost/Upgrade: {cost_per_upgrade:,.0f}
"""
        
        ax.text(0.5, 0.5, stats_text, fontsize=16, fontweight='bold', 
               color=colors['gold'], verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle='round,pad=1', facecolor=colors['neutral'], alpha=0.3, edgecolor=colors['gold'], linewidth=3))
    
    def _plot_success_rates_info(self, ax, colors):
        """Plot success rates as info box"""
        ax.axis('off')
        
        # Calculate success rates
        rates_text = "SUCCESS RATES\n\n"
        for level_key, level_name in [('3_to_4', '+3→+4'), ('4_to_5', '+4→+5'), 
                                       ('5_to_6', '+5→+6'), ('6_to_7', '+6→+7')]:
            data = self.total_upgrade_data[level_key]
            if data['attempts'] > 0:
                rate = (data['successes'] / data['attempts']) * 100
            else:
                rate = 0
            rates_text += f"{level_name}: {rate:.1f}%\n"
        
        ax.text(0.5, 0.5, rates_text, fontsize=16, fontweight='bold', 
               color=colors['success'], verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle='round,pad=1', facecolor=colors['neutral'], alpha=0.3, edgecolor=colors['gold'], linewidth=3))
    
    def _plot_success_failure_bars(self, ax, colors):
        """Plot success vs failure vs total attempts distribution"""
        levels = ['+3→+4', '+4→+5', '+5→+6', '+6→+7']
        successes = [self.total_upgrade_data[level]['successes'] 
                    for level in ['3_to_4', '4_to_5', '5_to_6', '6_to_7']]
        failures = [self.total_upgrade_data[level]['failures'] 
                   for level in ['3_to_4', '4_to_5', '5_to_6', '6_to_7']]
        attempts = [self.total_upgrade_data[level]['attempts'] 
                   for level in ['3_to_4', '4_to_5', '5_to_6', '6_to_7']]
        
        x = np.arange(len(levels))
        width = 0.25
        
        bars1 = ax.bar(x - width, attempts, width, label='Total Attempts', 
                      color=colors['neutral'], alpha=0.8, edgecolor=colors['gold'], linewidth=2)
        bars2 = ax.bar(x, successes, width, label='Success', 
                      color=colors['success'], alpha=0.8, edgecolor=colors['gold'], linewidth=2)
        bars3 = ax.bar(x + width, failures, width, label='Failure', 
                      color=colors['failure'], alpha=0.8, edgecolor=colors['gold'], linewidth=2)
        
        # Add count labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')
        
        ax.set_xlabel('Upgrade Level', fontsize=18, fontweight='bold', color=colors['gold'])
        ax.set_ylabel('Count', fontsize=18, fontweight='bold', color=colors['gold'])
        ax.set_title('Total Attempts / Success / Failure by Upgrade Level', fontsize=20, fontweight='bold', color=colors['neutral'])
        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14, colors='white')
        ax.legend(fontsize=14, loc='upper right', framealpha=0.8)
        ax.grid(axis='y', alpha=0.3, linestyle='--', color=colors['neutral'])


def main():
    """Main function"""
    print("🎨 Knight Online Upgrade Visualizer")
    print("=" * 60)
    
    visualizer = UpgradeVisualizer()
    
    if not visualizer.load_all_data():
        print("❌ Failed to load upgrade data!")
        return
    
    print("\n🎨 Creating visualization...")
    filename = visualizer.create_visualization()
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Saved as: {filename}")


if __name__ == "__main__":
    main()