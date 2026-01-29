# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Distilled logical shape training examples.

90 examples (15 per shape) that teach fundamental reasoning patterns.
Each example explicitly identifies the logical shape and walks through
the reasoning step-by-step.

Research Finding:
    10 perfect examples > 10,000 mediocre ones.
    Quality training data with explicit shape identification produces
    better generalization than large quantities of surface-level examples.
"""

from __future__ import annotations

from modelcypher.core.domain.training.logical_shapes import (
    LogicalShape,
    LogicalShapeExample,
)


def get_logical_shape_examples() -> list[LogicalShapeExample]:
    """Get all distilled logical shape training examples.

    Returns:
        List of 90 LogicalShapeExample objects (15 per shape).
    """
    return (
        _percentage_increase_examples()
        + _average_rate_examples()
        + _threshold_crossing_examples()
        + _inverse_chain_examples()
        + _sequential_operations_examples()
        + _remaining_first_examples()
    )


def _percentage_increase_examples() -> list[LogicalShapeExample]:
    """15 examples for PERCENTAGE_INCREASE shape."""
    return [
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A house worth $100 increases in value by 50%. What is it worth now?",
            reasoning=(
                '"Increased by 50%" means we ADD 50% of the original to the original.\n\n'
                "Step 1: What is 50% of $100?\n"
                "50% of $100 = $100 × 0.50 = $50\n\n"
                "Step 2: Add this increase to the original:\n"
                "New value = $100 + $50 = $150\n\n"
                "The house is worth $150."
            ),
            answer="150",
            common_error="Forgetting to add to original: $100 × 0.50 = $50 (wrong)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A painting worth $200 increases in value by 100%. What is the new value?",
            reasoning=(
                '"Increased by 100%" means we ADD 100% of the original (the full original amount).\n\n'
                "Step 1: 100% of $200 = $200\n"
                "Step 2: New value = $200 + $200 = $400\n\n"
                "(Note: Increase by 100% = double the value)"
            ),
            answer="400",
            common_error="Thinking 100% increase means just the original ($200)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A car bought for $80,000 increases in value by 150% after restoration. What is its new value?",
            reasoning=(
                'CRITICAL: "Increased by 150%" is NOT "multiply by 1.5" (that\'s only 50% increase).\n'
                '"Increased by 150%" means ADD 150% of the original.\n\n'
                "Step 1: 150% of $80,000 = $80,000 × 1.5 = $120,000\n"
                "Step 2: New value = Original + Increase = $80,000 + $120,000 = $200,000"
            ),
            answer="200000",
            common_error="Multiplying by 1.5 gives $120,000 (50% increase, not 150%)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="Someone buys a house for $80,000, spends $50,000 on repairs. The house value increases by 150% from the purchase price. What is their profit?",
            reasoning=(
                "Step 1: Calculate new value using percentage increase rule.\n"
                '"Increased by 150%" means: New = Original + (Original × 1.5)\n'
                "New value = $80,000 + ($80,000 × 1.5) = $80,000 + $120,000 = $200,000\n\n"
                "Step 2: Calculate total cost.\n"
                "Total cost = Purchase + Repairs = $80,000 + $50,000 = $130,000\n\n"
                "Step 3: Calculate profit.\n"
                "Profit = New value - Total cost = $200,000 - $130,000 = $70,000"
            ),
            answer="70000",
            common_error="Using wrong percentage formula or forgetting repair costs",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="Stock worth $500 increases by 200%. New value?",
            reasoning=(
                "Step 1: Calculate the increase amount.\n"
                "200% of $500 = $500 × 2.0 = $1000\n\n"
                "Step 2: Add to original.\n"
                "New = $500 + $1000 = $1500"
            ),
            answer="1500",
            common_error="Multiplying by 2.0 gives $1000 (100% increase, not 200%)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A salary of $50,000 increases by 10%. What is the new salary?",
            reasoning=(
                "Step 1: Calculate the increase.\n"
                "10% of $50,000 = $50,000 × 0.10 = $5,000\n\n"
                "Step 2: Add to original.\n"
                "New salary = $50,000 + $5,000 = $55,000"
            ),
            answer="55000",
            common_error="None common for simple percentages, but always add to original",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A population of 1,000 increases by 25%. New population?",
            reasoning=(
                "Step 1: Calculate increase.\n"
                "25% of 1,000 = 1,000 × 0.25 = 250\n\n"
                "Step 2: Add to original.\n"
                "New population = 1,000 + 250 = 1,250"
            ),
            answer="1250",
            common_error="Forgetting this is about people (whole numbers)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A $60 item increases in price by 75%. New price?",
            reasoning=(
                "Step 1: Calculate increase.\n"
                "75% of $60 = $60 × 0.75 = $45\n\n"
                "Step 2: Add to original.\n"
                "New price = $60 + $45 = $105"
            ),
            answer="105",
            common_error="Multiplying by 0.75 gives $45 (the increase only)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="An investment of $10,000 increases by 8% annually. Value after 1 year?",
            reasoning=(
                "Step 1: Calculate increase.\n"
                "8% of $10,000 = $10,000 × 0.08 = $800\n\n"
                "Step 2: Add to original.\n"
                "Value after 1 year = $10,000 + $800 = $10,800"
            ),
            answer="10800",
            common_error="Forgetting to add the interest to principal",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A $2,000 laptop increases in value by 5% due to collector demand. New value?",
            reasoning=(
                "Step 1: Calculate increase.\n"
                "5% of $2,000 = $2,000 × 0.05 = $100\n\n"
                "Step 2: Add to original.\n"
                "New value = $2,000 + $100 = $2,100"
            ),
            answer="2100",
            common_error="Small percentages still require the same formula",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A store's sales of $40,000 increased by 35%. What are the new sales?",
            reasoning=(
                "Step 1: Calculate increase.\n"
                "35% of $40,000 = $40,000 × 0.35 = $14,000\n\n"
                "Step 2: Add to original.\n"
                "New sales = $40,000 + $14,000 = $54,000"
            ),
            answer="54000",
            common_error="Using 1.35 directly works here: $40,000 × 1.35 = $54,000",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A company's revenue of $1 million increased by 300%. New revenue?",
            reasoning=(
                "Step 1: Calculate increase.\n"
                "300% of $1,000,000 = $1,000,000 × 3.0 = $3,000,000\n\n"
                "Step 2: Add to original.\n"
                "New revenue = $1,000,000 + $3,000,000 = $4,000,000"
            ),
            answer="4000000",
            common_error="Multiplying by 3 gives $3M (200% increase, not 300%)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="Temperature rises from 20°C by 15%. New temperature?",
            reasoning=(
                "Step 1: Calculate increase.\n"
                "15% of 20 = 20 × 0.15 = 3\n\n"
                "Step 2: Add to original.\n"
                "New temperature = 20 + 3 = 23°C"
            ),
            answer="23",
            common_error="Percentage of temperature works the same as other quantities",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A 50kg weight increases by 20%. New weight?",
            reasoning=(
                "Step 1: Calculate increase.\n"
                "20% of 50 = 50 × 0.20 = 10\n\n"
                "Step 2: Add to original.\n"
                "New weight = 50 + 10 = 60kg"
            ),
            answer="60",
            common_error="Simple application of the percentage increase formula",
        ),
        LogicalShapeExample(
            shape=LogicalShape.PERCENTAGE_INCREASE,
            question="A website has 8,000 users. Membership increased by 125%. How many users now?",
            reasoning=(
                "Step 1: Calculate increase.\n"
                "125% of 8,000 = 8,000 × 1.25 = 10,000\n\n"
                "Step 2: Add to original.\n"
                "New users = 8,000 + 10,000 = 18,000"
            ),
            answer="18000",
            common_error="Multiplying by 1.25 gives 10,000 (only 25% increase)",
        ),
    ]


def _average_rate_examples() -> list[LogicalShapeExample]:
    """15 examples for AVERAGE_RATE shape."""
    return [
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A car travels 60 miles at 30 mph, then 60 miles at 60 mph. What is the average speed?",
            reasoning=(
                "CRITICAL: Average speed is NOT (30 + 60) / 2 = 45 mph. That's WRONG!\n"
                "Average speed = Total distance / Total time\n\n"
                "Step 1: Calculate time for each segment.\n"
                "Time at 30 mph: 60 miles ÷ 30 mph = 2 hours\n"
                "Time at 60 mph: 60 miles ÷ 60 mph = 1 hour\n\n"
                "Step 2: Calculate totals.\n"
                "Total distance = 60 + 60 = 120 miles\n"
                "Total time = 2 + 1 = 3 hours\n\n"
                "Step 3: Calculate average speed.\n"
                "Average speed = 120 miles / 3 hours = 40 mph"
            ),
            answer="40",
            common_error="(30 + 60) / 2 = 45 mph (arithmetic mean is wrong)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="John drives 180 miles to a destination at 60 mph. He returns at 30 mph due to traffic. What is his average speed for the round trip?",
            reasoning=(
                "Step 1: Time going = 180 ÷ 60 = 3 hours\n"
                "Step 2: Time returning = 180 ÷ 30 = 6 hours\n"
                "Step 3: Total distance = 180 + 180 = 360 miles\n"
                "Step 4: Total time = 3 + 6 = 9 hours\n"
                "Step 5: Average speed = 360 / 9 = 40 mph\n\n"
                "(Note: NOT (60+30)/2 = 45. The slower speed is weighted more because more TIME is spent at that speed.)"
            ),
            answer="40",
            common_error="(60 + 30) / 2 = 45 mph (ignores time weighting)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A runner jogs 5 miles at 5 mph, then runs 5 miles at 10 mph. What is the average speed?",
            reasoning=(
                "Step 1: Time jogging: 5 ÷ 5 = 1 hour\n"
                "Step 2: Time running: 5 ÷ 10 = 0.5 hours\n"
                "Step 3: Total distance: 10 miles\n"
                "Step 4: Total time: 1.5 hours\n"
                "Step 5: Average speed: 10 / 1.5 = 6.67 mph ≈ 7 mph"
            ),
            answer="7",
            common_error="(5 + 10) / 2 = 7.5 mph (happens to be close but wrong method)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A cyclist rides 20 miles at 20 mph, then 20 miles at 10 mph. Average speed?",
            reasoning=(
                "Step 1: Time at 20 mph: 20 ÷ 20 = 1 hour\n"
                "Step 2: Time at 10 mph: 20 ÷ 10 = 2 hours\n"
                "Step 3: Total distance: 40 miles\n"
                "Step 4: Total time: 3 hours\n"
                "Step 5: Average speed: 40 / 3 = 13.33 mph ≈ 13 mph"
            ),
            answer="13",
            common_error="(20 + 10) / 2 = 15 mph (ignores time)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A hiker walks 8 miles at 4 mph uphill, then 8 miles at 8 mph downhill. Average speed?",
            reasoning=(
                "Step 1: Time uphill: 8 ÷ 4 = 2 hours\n"
                "Step 2: Time downhill: 8 ÷ 8 = 1 hour\n"
                "Step 3: Total distance: 16 miles\n"
                "Step 4: Total time: 3 hours\n"
                "Step 5: Average speed: 16 / 3 = 5.33 mph ≈ 5 mph"
            ),
            answer="5",
            common_error="(4 + 8) / 2 = 6 mph (arithmetic mean)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A swimmer covers 100m at 2 m/s, then 100m at 1 m/s. Average speed?",
            reasoning=(
                "Step 1: Time at 2 m/s: 100 ÷ 2 = 50 seconds\n"
                "Step 2: Time at 1 m/s: 100 ÷ 1 = 100 seconds\n"
                "Step 3: Total distance: 200m\n"
                "Step 4: Total time: 150 seconds\n"
                "Step 5: Average speed: 200 / 150 = 1.33 m/s ≈ 1 m/s"
            ),
            answer="1",
            common_error="(2 + 1) / 2 = 1.5 m/s (wrong)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A plane flies 600 miles at 600 mph, then 600 miles at 300 mph. Average speed?",
            reasoning=(
                "Step 1: Time at 600 mph: 600 ÷ 600 = 1 hour\n"
                "Step 2: Time at 300 mph: 600 ÷ 300 = 2 hours\n"
                "Step 3: Total distance: 1200 miles\n"
                "Step 4: Total time: 3 hours\n"
                "Step 5: Average speed: 1200 / 3 = 400 mph"
            ),
            answer="400",
            common_error="(600 + 300) / 2 = 450 mph",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A car drives 100 km at 50 km/h, then 200 km at 100 km/h. Average speed?",
            reasoning=(
                "Step 1: Time at 50 km/h: 100 ÷ 50 = 2 hours\n"
                "Step 2: Time at 100 km/h: 200 ÷ 100 = 2 hours\n"
                "Step 3: Total distance: 300 km\n"
                "Step 4: Total time: 4 hours\n"
                "Step 5: Average speed: 300 / 4 = 75 km/h"
            ),
            answer="75",
            common_error="(50 + 100) / 2 = 75 (happens to match but method is wrong)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A truck hauls 50 loads at 25 mph, then 50 loads at 50 mph. Average speed?",
            reasoning=(
                "Assuming each 'load' covers the same distance:\n"
                "Step 1: Time at 25 mph: d/25 per load × 50 = 50d/25 = 2d\n"
                "Step 2: Time at 50 mph: d/50 per load × 50 = 50d/50 = d\n"
                "Step 3: Total distance: 100d\n"
                "Step 4: Total time: 3d\n"
                "Step 5: Average speed: 100d / 3d = 33.3 mph ≈ 33 mph"
            ),
            answer="33",
            common_error="(25 + 50) / 2 = 37.5 mph",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="Worker A completes a job in 2 hours, Worker B in 3 hours. Together, what's their combined rate per hour?",
            reasoning=(
                "Step 1: Rate of A: 1 job / 2 hours = 0.5 jobs/hour\n"
                "Step 2: Rate of B: 1 job / 3 hours = 0.333 jobs/hour\n"
                "Step 3: Combined rate: 0.5 + 0.333 = 0.833 jobs/hour\n"
                "Step 4: Time together: 1 / 0.833 = 1.2 hours\n\n"
                "Combined rate is 0.833 jobs/hour, or they complete 1 job in 1.2 hours."
            ),
            answer="1",
            common_error="Averaging hours: (2+3)/2 = 2.5 hours (wrong concept)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A boat travels 30 miles downstream at 15 mph, then 30 miles upstream at 5 mph. Average speed?",
            reasoning=(
                "Step 1: Time downstream: 30 ÷ 15 = 2 hours\n"
                "Step 2: Time upstream: 30 ÷ 5 = 6 hours\n"
                "Step 3: Total distance: 60 miles\n"
                "Step 4: Total time: 8 hours\n"
                "Step 5: Average speed: 60 / 8 = 7.5 mph ≈ 8 mph"
            ),
            answer="8",
            common_error="(15 + 5) / 2 = 10 mph",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A delivery person drives 10 miles at 40 mph to a location, waits 15 minutes, then drives back at 60 mph. Average speed for the trip?",
            reasoning=(
                "Step 1: Time driving there: 10 ÷ 40 = 0.25 hours = 15 min\n"
                "Step 2: Wait time: 15 min = 0.25 hours\n"
                "Step 3: Time driving back: 10 ÷ 60 = 0.167 hours = 10 min\n"
                "Step 4: Total distance: 20 miles\n"
                "Step 5: Total time: 0.25 + 0.25 + 0.167 = 0.667 hours = 40 min\n"
                "Step 6: Average speed: 20 / 0.667 = 30 mph"
            ),
            answer="30",
            common_error="Forgetting wait time or averaging speeds directly",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A train covers the first half of a journey at 40 km/h and the second half at 60 km/h. Average speed?",
            reasoning=(
                "Let total distance = 2d (each half = d)\n"
                "Step 1: Time for first half: d ÷ 40\n"
                "Step 2: Time for second half: d ÷ 60\n"
                "Step 3: Total time: d/40 + d/60 = 3d/120 + 2d/120 = 5d/120 = d/24\n"
                "Step 4: Average speed: 2d ÷ (5d/120) = 2d × 120/5d = 240/5 = 48 km/h"
            ),
            answer="48",
            common_error="(40 + 60) / 2 = 50 km/h",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="A machine produces 100 units in the first hour, then 200 units in the next 2 hours. Average rate?",
            reasoning=(
                "Step 1: Total units: 100 + 200 = 300 units\n"
                "Step 2: Total time: 1 + 2 = 3 hours\n"
                "Step 3: Average rate: 300 / 3 = 100 units/hour"
            ),
            answer="100",
            common_error="Averaging rates: (100 + 100) / 2 = 100 (coincidentally correct)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.AVERAGE_RATE,
            question="Data transfers at 10 MB/s for 100 MB, then 5 MB/s for 100 MB. Average transfer speed?",
            reasoning=(
                "Step 1: Time at 10 MB/s: 100 ÷ 10 = 10 seconds\n"
                "Step 2: Time at 5 MB/s: 100 ÷ 5 = 20 seconds\n"
                "Step 3: Total data: 200 MB\n"
                "Step 4: Total time: 30 seconds\n"
                "Step 5: Average speed: 200 / 30 = 6.67 MB/s ≈ 7 MB/s"
            ),
            answer="7",
            common_error="(10 + 5) / 2 = 7.5 MB/s",
        ),
    ]


def _threshold_crossing_examples() -> list[LogicalShapeExample]:
    """15 examples for THRESHOLD_CROSSING shape."""
    return [
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="A tree costs $100 to plant. It produces $10 of fruit per year. In what year do you START making money?",
            reasoning=(
                "Step 1: Find breakeven point.\n"
                "Years to break even = $100 / $10 per year = 10 years exactly\n\n"
                'Step 2: Determine "START making money."\n'
                "After year 10: You've earned exactly $100, which equals your cost. Net = $0.\n"
                "Year 10 is breakeven, not profit.\n\n"
                "After year 11: You've earned $110. Net = +$10. This is your first profit!\n\n"
                "Year 11 is when you START making money."
            ),
            answer="11",
            common_error="Reporting year 10 (breakeven) instead of year 11 (first profit)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="An investment costs $90. It returns $7.50 per year. In which year do you first have positive returns?",
            reasoning=(
                "Step 1: Exact breakeven = $90 / $7.50 = 12 years\n\n"
                "Step 2: Since 12 × $7.50 = $90 exactly, year 12 is exact breakeven (zero profit).\n\n"
                "Year 13 is when cumulative returns ($97.50) first EXCEED the cost ($90).\n"
                "Year 13 is when you START earning."
            ),
            answer="13",
            common_error="Reporting 12 (breakeven)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="A lemon tree costs $90 to plant. It produces $10.50 of lemons per year but costs $3 per year to maintain. When does the owner start earning money?",
            reasoning=(
                "Step 1: Net annual profit = $10.50 - $3 = $7.50\n"
                "Step 2: Breakeven = $90 / $7.50 = 12 years exactly\n"
                "Step 3: Year 12 is breakeven (12 × $7.50 = $90 = cost)\n"
                "Step 4: Year 13 is first year with positive cumulative profit"
            ),
            answer="13",
            common_error="Forgetting maintenance costs or reporting breakeven year",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="Machine costs $60, earns $5/month. When do you profit?",
            reasoning=(
                "Breakeven: 60/5 = 12 months exactly.\n"
                "First profit: month 13."
            ),
            answer="13",
            common_error="Reporting month 12",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="Solar panels cost $5,000. They save $500/year on electricity. When do savings exceed cost?",
            reasoning=(
                "Step 1: Years to breakeven = $5,000 / $500 = 10 years\n"
                "Step 2: At year 10: savings = cost (breakeven)\n"
                "Step 3: At year 11: savings = $5,500 > $5,000 cost\n"
                "First year savings EXCEED cost is year 11."
            ),
            answer="11",
            common_error="Reporting year 10",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="A gym membership costs $240/year. Personal training sessions cost $20 each. After how many sessions does the membership become worthwhile (sessions normally $30)?",
            reasoning=(
                "Step 1: Savings per session = $30 - $20 = $10\n"
                "Step 2: Breakeven sessions = $240 / $10 = 24 sessions\n"
                "Step 3: At 24 sessions: savings = membership cost (breakeven)\n"
                "Step 4: At 25 sessions: you START saving money"
            ),
            answer="25",
            common_error="Reporting 24 sessions",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="A car costs $20,000 but saves $200/month in commuting. When is the car 'paid for' by savings?",
            reasoning=(
                "Step 1: Months to breakeven = $20,000 / $200 = 100 months\n"
                "Step 2: At month 100: savings = cost\n"
                "Step 3: At month 101: cumulative savings exceed cost"
            ),
            answer="101",
            common_error="Reporting month 100",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="A business invests $12,000. Monthly revenue is $1,000 and costs are $500. When is the business profitable overall?",
            reasoning=(
                "Step 1: Monthly profit = $1,000 - $500 = $500\n"
                "Step 2: Breakeven months = $12,000 / $500 = 24 months\n"
                "Step 3: Month 24 is breakeven\n"
                "Step 4: Month 25 is first overall profit"
            ),
            answer="25",
            common_error="Reporting month 24",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="An orchard costs $8,000 to establish. Fruit sales yield $2,000/year after $400/year in costs. When does it become profitable?",
            reasoning=(
                "Step 1: Net annual profit = $2,000 - $400 = $1,600\n"
                "Step 2: Years to breakeven = $8,000 / $1,600 = 5 years exactly\n"
                "Step 3: Year 5 is breakeven\n"
                "Step 4: Year 6 is first profit"
            ),
            answer="6",
            common_error="Reporting year 5",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="A factory upgrade costs $150,000 and saves $15,000/year. When do total savings exceed the investment?",
            reasoning=(
                "Step 1: Years to breakeven = $150,000 / $15,000 = 10 years\n"
                "Step 2: Year 10: savings = investment (breakeven)\n"
                "Step 3: Year 11: savings = $165,000 > $150,000"
            ),
            answer="11",
            common_error="Reporting year 10",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="A coffee maker costs $120. Making coffee at home saves $3/day vs buying. When do you start saving?",
            reasoning=(
                "Step 1: Days to breakeven = $120 / $3 = 40 days\n"
                "Step 2: Day 40 is breakeven\n"
                "Step 3: Day 41 is first day you're actually saving"
            ),
            answer="41",
            common_error="Reporting day 40",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="Education costs $40,000 but increases salary by $5,000/year. When does it pay off?",
            reasoning=(
                "Step 1: Years to breakeven = $40,000 / $5,000 = 8 years\n"
                "Step 2: Year 8 is breakeven\n"
                "Step 3: Year 9 is first year the education has paid off"
            ),
            answer="9",
            common_error="Reporting year 8",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="A beehive costs $800 and produces $80 worth of honey per year. When is the first profitable year?",
            reasoning=(
                "Step 1: Years to breakeven = $800 / $80 = 10 years exactly\n"
                "Step 2: Year 10: revenue = cost (breakeven)\n"
                "Step 3: Year 11 is first profitable year"
            ),
            answer="11",
            common_error="Reporting year 10",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="You buy a $600 pass. Individual tickets cost $15. After how many uses do you START saving?",
            reasoning=(
                "Step 1: Uses to breakeven = $600 / $15 = 40 uses\n"
                "Step 2: At 40 uses: pass cost = ticket cost (breakeven)\n"
                "Step 3: At 41 uses: you START saving"
            ),
            answer="41",
            common_error="Reporting 40 uses",
        ),
        LogicalShapeExample(
            shape=LogicalShape.THRESHOLD_CROSSING,
            question="A franchise fee is $50,000. Monthly profit is $2,500. When is the franchise 'in the black'?",
            reasoning=(
                "Step 1: Months to breakeven = $50,000 / $2,500 = 20 months\n"
                "Step 2: Month 20: cumulative profit = franchise fee\n"
                "Step 3: Month 21: first month 'in the black'"
            ),
            answer="21",
            common_error="Reporting month 20",
        ),
    ]


def _inverse_chain_examples() -> list[LogicalShapeExample]:
    """15 examples for INVERSE_CHAIN shape."""
    return [
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="After giving away half of her apples, Sarah has 5 left. How many did she start with?",
            reasoning=(
                "Work BACKWARDS from what we know.\n\n"
                "We know: After giving half, she has 5.\n"
                "Inverse: If 5 is half, then whole = 5 × 2 = 10\n\n"
                "She started with 10 apples."
            ),
            answer="10",
            common_error="Working forwards: 5 ÷ 2 = 2.5 (wrong direction)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="A salesperson sold 1/3 of their items at the first house, then 2 more items at the second house, and half of what remained at the third house. They sold 5 items at the third house. How many items did they start with?",
            reasoning=(
                "Work BACKWARDS from the known endpoint.\n\n"
                "Step 1: At third house, sold 5, which was HALF of remaining.\n"
                "So before third house: 5 × 2 = 10 items\n\n"
                "Step 2: At second house, sold 2 items.\n"
                "So before second house: 10 + 2 = 12 items\n\n"
                "Step 3: After first house, had 12 items. This is 2/3 of original (since 1/3 was sold).\n"
                "Original = 12 × (3/2) = 18 items\n\n"
                "Verify: Start 18 → sell 6 (1/3) → 12 left → sell 2 → 10 left → sell 5 (half) → 5 left ✓"
            ),
            answer="18",
            common_error="Working forwards with unknowns leads to confusion",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="Half of Tom's remaining money is $15. How much does he have?",
            reasoning=(
                "If half is $15, then whole is $15 × 2 = $30."
            ),
            answer="30",
            common_error="Dividing: $15 ÷ 2 = $7.50 (wrong operation)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="A third of the cookies were eaten, leaving 20. How many were there originally?",
            reasoning=(
                "20 is 2/3 of original (since 1/3 was eaten).\n"
                "Original = 20 × (3/2) = 30"
            ),
            answer="30",
            common_error="20 × 3 = 60 (wrong - that would be if 20 was 1/3)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="After tripling a number, you get 27. What was the original number?",
            reasoning=(
                "Work backwards: If tripling gives 27, then original = 27 ÷ 3 = 9"
            ),
            answer="9",
            common_error="27 × 3 = 81 (going forward instead of backward)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="You added 7 to a number, then doubled it to get 30. What was the original number?",
            reasoning=(
                "Work backwards through operations:\n"
                "Step 1: Before doubling: 30 ÷ 2 = 15\n"
                "Step 2: Before adding 7: 15 - 7 = 8\n\n"
                "Verify: 8 + 7 = 15, 15 × 2 = 30 ✓"
            ),
            answer="8",
            common_error="Wrong order: (30 - 7) ÷ 2 = 11.5",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="After spending 1/4 of her money on lunch and $10 more on a book, Jane has $15 left. How much did she start with?",
            reasoning=(
                "Work backwards:\n"
                "Step 1: Before buying book: $15 + $10 = $25\n"
                "Step 2: $25 is 3/4 of original (since 1/4 was spent on lunch)\n"
                "Step 3: Original = $25 × (4/3) = $33.33\n\n"
                "Actually, let's check: 33.33 × (1/4) = 8.33 for lunch, leaving 25. Then -10 = 15. ✓"
            ),
            answer="33",
            common_error="Incorrect fraction handling when working backwards",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="A number was multiplied by 4, then 8 was subtracted to get 20. Find the original number.",
            reasoning=(
                "Work backwards:\n"
                "Step 1: Before subtracting 8: 20 + 8 = 28\n"
                "Step 2: Before multiplying by 4: 28 ÷ 4 = 7\n\n"
                "Verify: 7 × 4 = 28, 28 - 8 = 20 ✓"
            ),
            answer="7",
            common_error="(20 + 8) × 4 = 112 (wrong order)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="After losing 1/5 of his marbles to Jim and giving 6 to Sam, Bob has 14 left. How many did Bob start with?",
            reasoning=(
                "Work backwards:\n"
                "Step 1: Before giving 6 to Sam: 14 + 6 = 20\n"
                "Step 2: 20 is 4/5 of original (since 1/5 was lost)\n"
                "Step 3: Original = 20 × (5/4) = 25"
            ),
            answer="25",
            common_error="Applying fractions in wrong order",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="A store reduced prices by 20%, then reduced the sale price by another 10%. The final price is $72. What was the original price?",
            reasoning=(
                "Work backwards:\n"
                "Step 1: $72 is 90% of sale price. Sale price = $72 / 0.9 = $80\n"
                "Step 2: $80 is 80% of original. Original = $80 / 0.8 = $100\n\n"
                "Verify: $100 × 0.8 = $80, $80 × 0.9 = $72 ✓"
            ),
            answer="100",
            common_error="$72 / 0.7 = $102.86 (combining percentages wrong)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="After paying 15% tax, the total bill is $115. What was the pre-tax amount?",
            reasoning=(
                "$115 is 115% of original (100% + 15% tax)\n"
                "Original = $115 / 1.15 = $100"
            ),
            answer="100",
            common_error="$115 × 0.85 = $97.75 (subtracting instead of dividing)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="You took half of a pile of candies, then took 3 more. You ended up with 10 candies. How many were in the pile?",
            reasoning=(
                "Work backwards:\n"
                "Step 1: Before taking 3 more: 10 - 3 = 7\n"
                "Step 2: 7 is half of original pile. Original = 7 × 2 = 14"
            ),
            answer="14",
            common_error="(10 - 3) × 2 = 14 ✓ (this one is straightforward)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="A length was increased by 5cm, then doubled, resulting in 30cm. What was the original length?",
            reasoning=(
                "Work backwards:\n"
                "Step 1: Before doubling: 30 ÷ 2 = 15cm\n"
                "Step 2: Before adding 5cm: 15 - 5 = 10cm\n\n"
                "Verify: 10 + 5 = 15, 15 × 2 = 30 ✓"
            ),
            answer="10",
            common_error="(30 ÷ 2) - 5 = 10 (correct but easy to mix up)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="After 3 equal deposits, a bank account has $180. Before the deposits it had $60. How much was each deposit?",
            reasoning=(
                "Work backwards:\n"
                "Step 1: Total deposits = $180 - $60 = $120\n"
                "Step 2: Each deposit = $120 ÷ 3 = $40"
            ),
            answer="40",
            common_error="$180 ÷ 3 = $60 (forgetting initial balance)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.INVERSE_CHAIN,
            question="A number was squared, then 9 was added, resulting in 25. What was the original number?",
            reasoning=(
                "Work backwards:\n"
                "Step 1: Before adding 9: 25 - 9 = 16\n"
                "Step 2: Before squaring: √16 = 4\n\n"
                "Verify: 4² = 16, 16 + 9 = 25 ✓"
            ),
            answer="4",
            common_error="√(25 - 9) = √16 = 4 (correct)",
        ),
    ]


def _sequential_operations_examples() -> list[LogicalShapeExample]:
    """15 examples for SEQUENTIAL_OPERATIONS shape."""
    return [
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="A farmer has 20 eggs. She uses 5 for breakfast and 3 for baking. She sells the rest at $2 each. How much does she make?",
            reasoning=(
                "Step 1: Calculate remaining (subtract FIRST).\n"
                "Remaining = 20 - 5 - 3 = 12 eggs\n\n"
                "Step 2: Calculate revenue (multiply SECOND).\n"
                "Revenue = 12 × $2 = $24"
            ),
            answer="24",
            common_error="20 × $2 - 5 - 3 = $32 (wrong order)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for baking muffins. She sells the rest at $2 each. How much does she make daily?",
            reasoning=(
                "Step 1: Eggs used = 3 + 4 = 7\n"
                "Step 2: Eggs remaining = 16 - 7 = 9\n"
                "Step 3: Daily revenue = 9 × $2 = $18"
            ),
            answer="18",
            common_error="16 × $2 = $32 (ignoring eggs used)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="A box has 100 items. 25 are defective and removed. The rest are packaged in groups of 5. How many packages?",
            reasoning=(
                "Step 1: Remove defective (subtract FIRST).\n"
                "Good items = 100 - 25 = 75\n\n"
                "Step 2: Package (divide SECOND).\n"
                "Packages = 75 ÷ 5 = 15"
            ),
            answer="15",
            common_error="100 ÷ 5 = 20 (ignoring defective items)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="A store has 50 shirts. 8 are on display. The rest are priced at $15 each. Total value of undisplayed shirts?",
            reasoning=(
                "Step 1: Undisplayed shirts = 50 - 8 = 42\n"
                "Step 2: Total value = 42 × $15 = $630"
            ),
            answer="630",
            common_error="50 × $15 = $750 (including displayed shirts)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="You have $100. You spend $30 on lunch and $15 on coffee. You divide the rest equally among 5 friends. How much does each friend get?",
            reasoning=(
                "Step 1: Remaining after spending = $100 - $30 - $15 = $55\n"
                "Step 2: Per friend = $55 ÷ 5 = $11"
            ),
            answer="11",
            common_error="$100 ÷ 5 = $20 (ignoring expenses)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="A baker makes 60 cookies. She sets aside 12 for her family. The rest are sold in boxes of 6. How many boxes?",
            reasoning=(
                "Step 1: Cookies for sale = 60 - 12 = 48\n"
                "Step 2: Boxes = 48 ÷ 6 = 8"
            ),
            answer="8",
            common_error="60 ÷ 6 = 10 (ignoring family portion)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="A tank holds 200 gallons. 50 gallons leak out. The rest is distributed among 10 containers. Gallons per container?",
            reasoning=(
                "Step 1: Remaining = 200 - 50 = 150 gallons\n"
                "Step 2: Per container = 150 ÷ 10 = 15 gallons"
            ),
            answer="15",
            common_error="200 ÷ 10 = 20 (ignoring leak)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="A class has 30 students. 6 are absent. The present students form groups of 4. How many full groups?",
            reasoning=(
                "Step 1: Present = 30 - 6 = 24\n"
                "Step 2: Full groups = 24 ÷ 4 = 6"
            ),
            answer="6",
            common_error="30 ÷ 4 = 7.5 (ignoring absences)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="You earn $500. After $120 in taxes and $80 in bills, you save half of the rest. How much do you save?",
            reasoning=(
                "Step 1: After deductions = $500 - $120 - $80 = $300\n"
                "Step 2: Savings = $300 × 0.5 = $150"
            ),
            answer="150",
            common_error="$500 × 0.5 = $250 (ignoring deductions)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="A tree has 80 apples. 15 fall off. Birds eat 10. The rest are picked and sold at $0.50 each. Total revenue?",
            reasoning=(
                "Step 1: Remaining = 80 - 15 - 10 = 55 apples\n"
                "Step 2: Revenue = 55 × $0.50 = $27.50"
            ),
            answer="28",
            common_error="80 × $0.50 = $40 (ignoring losses)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="A warehouse has 1000 items. 200 are shipped. The rest are sorted into bins of 40. How many bins?",
            reasoning=(
                "Step 1: Remaining = 1000 - 200 = 800\n"
                "Step 2: Bins = 800 ÷ 40 = 20"
            ),
            answer="20",
            common_error="1000 ÷ 40 = 25 (ignoring shipped items)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="You have 45 photos. You delete 9 blurry ones. The rest go into albums of 12 photos each. How many full albums?",
            reasoning=(
                "Step 1: Good photos = 45 - 9 = 36\n"
                "Step 2: Full albums = 36 ÷ 12 = 3"
            ),
            answer="3",
            common_error="45 ÷ 12 = 3.75 (ignoring deleted photos)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="A pizza has 12 slices. 3 are eaten immediately. The rest are divided among 3 friends. Slices per friend?",
            reasoning=(
                "Step 1: Remaining = 12 - 3 = 9 slices\n"
                "Step 2: Per friend = 9 ÷ 3 = 3 slices"
            ),
            answer="3",
            common_error="12 ÷ 3 = 4 (ignoring eaten slices)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="A garden has 90 plants. 15 die from frost. The rest are arranged in rows of 15. How many rows?",
            reasoning=(
                "Step 1: Surviving = 90 - 15 = 75 plants\n"
                "Step 2: Rows = 75 ÷ 15 = 5"
            ),
            answer="5",
            common_error="90 ÷ 15 = 6 (ignoring dead plants)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.SEQUENTIAL_OPERATIONS,
            question="You have 200 stickers. You give 50 to your sister and 30 to your brother. You arrange the rest in sheets of 20. How many sheets?",
            reasoning=(
                "Step 1: Remaining = 200 - 50 - 30 = 120 stickers\n"
                "Step 2: Sheets = 120 ÷ 20 = 6"
            ),
            answer="6",
            common_error="200 ÷ 20 = 10 (ignoring given stickers)",
        ),
    ]


def _remaining_first_examples() -> list[LogicalShapeExample]:
    """15 examples for REMAINING_FIRST shape."""
    return [
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A 12-mile trail. You walk 4 miles, then 2 miles. How many miles remain?",
            reasoning=(
                "Step 1: Calculate remaining.\n"
                "Remaining = Total - Part1 - Part2 = 12 - 4 - 2 = 6 miles"
            ),
            answer="6",
            common_error="Simple problem but sets up the pattern",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="Marissa is hiking a 12-mile trail. She walked 4 miles in the first hour, then 2 miles in the next 2 hours. If she finishes at 3 mph, how many more hours will it take?",
            reasoning=(
                "Step 1: Calculate remaining distance FIRST.\n"
                "Already walked = 4 + 2 = 6 miles\n"
                "Remaining = 12 - 6 = 6 miles\n\n"
                "Step 2: Calculate time at given rate.\n"
                "Time = Distance / Rate = 6 miles / 3 mph = 2 hours"
            ),
            answer="2",
            common_error="12 / 3 = 4 hours (using total distance)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="You need to read 100 pages. You read 30 pages on Monday and 25 on Tuesday. At 15 pages per hour, how many more hours do you need?",
            reasoning=(
                "Step 1: Remaining pages = 100 - 30 - 25 = 45 pages\n"
                "Step 2: Hours needed = 45 / 15 = 3 hours"
            ),
            answer="3",
            common_error="100 / 15 = 6.67 hours (using total pages)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A project requires 40 hours. You worked 8 hours Monday, 10 hours Tuesday. At 5 hours/day, how many more days?",
            reasoning=(
                "Step 1: Remaining hours = 40 - 8 - 10 = 22 hours\n"
                "Step 2: Days needed = 22 / 5 = 4.4 ≈ 5 days"
            ),
            answer="5",
            common_error="40 / 5 = 8 days (using total hours)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A tank needs 500 gallons. 200 gallons were added in the morning. If the pump adds 50 gallons/hour, how many more hours to fill?",
            reasoning=(
                "Step 1: Remaining = 500 - 200 = 300 gallons\n"
                "Step 2: Hours = 300 / 50 = 6 hours"
            ),
            answer="6",
            common_error="500 / 50 = 10 hours (using total capacity)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A marathon is 26 miles. A runner completed 18 miles. At 4 mph pace, how many more hours to finish?",
            reasoning=(
                "Step 1: Remaining = 26 - 18 = 8 miles\n"
                "Step 2: Hours = 8 / 4 = 2 hours"
            ),
            answer="2",
            common_error="26 / 4 = 6.5 hours (using total distance)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A goal is to save $1000. You've saved $350 so far. At $50/week, how many more weeks?",
            reasoning=(
                "Step 1: Remaining = $1000 - $350 = $650\n"
                "Step 2: Weeks = $650 / $50 = 13 weeks"
            ),
            answer="13",
            common_error="$1000 / $50 = 20 weeks (using total goal)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A download is 800 MB. 320 MB completed. At 40 MB/min, how many more minutes?",
            reasoning=(
                "Step 1: Remaining = 800 - 320 = 480 MB\n"
                "Step 2: Minutes = 480 / 40 = 12 minutes"
            ),
            answer="12",
            common_error="800 / 40 = 20 minutes (using total size)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A builder needs 200 bricks. He laid 75 yesterday and 45 today. At 20 bricks/hour, how many more hours?",
            reasoning=(
                "Step 1: Remaining = 200 - 75 - 45 = 80 bricks\n"
                "Step 2: Hours = 80 / 20 = 4 hours"
            ),
            answer="4",
            common_error="200 / 20 = 10 hours (using total bricks)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A pool holds 10,000 gallons. It has 6,000 gallons. A hose fills 200 gallons/hour. Hours to fill?",
            reasoning=(
                "Step 1: Remaining = 10,000 - 6,000 = 4,000 gallons\n"
                "Step 2: Hours = 4,000 / 200 = 20 hours"
            ),
            answer="20",
            common_error="10,000 / 200 = 50 hours (using total capacity)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A quilt needs 144 patches. 60 are sewn. At 12 patches/day, how many more days?",
            reasoning=(
                "Step 1: Remaining = 144 - 60 = 84 patches\n"
                "Step 2: Days = 84 / 12 = 7 days"
            ),
            answer="7",
            common_error="144 / 12 = 12 days (using total patches)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A novel has 300 pages. You've read 180. Reading 30 pages/hour, how many more hours?",
            reasoning=(
                "Step 1: Remaining = 300 - 180 = 120 pages\n"
                "Step 2: Hours = 120 / 30 = 4 hours"
            ),
            answer="4",
            common_error="300 / 30 = 10 hours (using total pages)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A car trip is 450 miles. After driving 290 miles, you stop. At 80 mph, how many more hours to destination?",
            reasoning=(
                "Step 1: Remaining = 450 - 290 = 160 miles\n"
                "Step 2: Hours = 160 / 80 = 2 hours"
            ),
            answer="2",
            common_error="450 / 80 = 5.625 hours (using total distance)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A fundraiser goal is $5,000. They raised $3,200. At $400/day, how many more days to reach the goal?",
            reasoning=(
                "Step 1: Remaining = $5,000 - $3,200 = $1,800\n"
                "Step 2: Days = $1,800 / $400 = 4.5 ≈ 5 days"
            ),
            answer="5",
            common_error="$5,000 / $400 = 12.5 days (using total goal)",
        ),
        LogicalShapeExample(
            shape=LogicalShape.REMAINING_FIRST,
            question="A recipe needs 24 cups of flour. You've measured 10 cups. Measuring 2 cups per minute, how many more minutes?",
            reasoning=(
                "Step 1: Remaining = 24 - 10 = 14 cups\n"
                "Step 2: Minutes = 14 / 2 = 7 minutes"
            ),
            answer="7",
            common_error="24 / 2 = 12 minutes (using total cups)",
        ),
    ]
