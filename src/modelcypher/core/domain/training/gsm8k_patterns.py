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

"""
GSM8K Pattern Training Data: Multi-step math reasoning.

Phase A of Project Polymath - targeting the 70% gap in GSM8K performance.

GSM8K patterns:
1. Sequential arithmetic chains
2. Percentage calculations
3. Rate/ratio problems
4. Unit conversions
5. Multi-step word problems with intermediate values
6. Comparison problems
7. Distribution/sharing problems
"""

from modelcypher.adapters.training.mlx.self_reflection import SelfReflectionExample


def get_gsm8k_pattern_examples() -> list[SelfReflectionExample]:
    """GSM8K-style multi-step math reasoning examples.

    These cover the main patterns found in GSM8K:
    - Sequential calculations (do X, then Y, then Z)
    - Intermediate value tracking
    - Rate problems
    - Percentage calculations
    - Distribution/sharing
    - Comparisons
    """
    return [
        # === SEQUENTIAL ARITHMETIC ===
        SelfReflectionExample(
            input_question="John has 5 apples. He buys 3 more, then gives away 2. How many apples does he have?",
            core_question="5 + 3 - 2 = ?",
            reasoning="Start: 5 apples\nBuys 3: 5 + 3 = 8\nGives 2: 8 - 2 = 6",
            answer="6",
        ),
        SelfReflectionExample(
            input_question="A store has 100 books. They sell 30 on Monday, 25 on Tuesday, and receive 50 new books on Wednesday. How many books do they have?",
            core_question="100 - 30 - 25 + 50 = ?",
            reasoning="Start: 100\nAfter Monday: 100 - 30 = 70\nAfter Tuesday: 70 - 25 = 45\nAfter Wednesday: 45 + 50 = 95",
            answer="95",
        ),
        SelfReflectionExample(
            input_question="Emma had $50. She spent $12 on lunch, $8 on coffee, and received $20 from a friend. How much money does she have now?",
            core_question="50 - 12 - 8 + 20 = ?",
            reasoning="Start: $50\nAfter lunch: 50 - 12 = 38\nAfter coffee: 38 - 8 = 30\nAfter gift: 30 + 20 = 50",
            answer="$50",
        ),

        # === MULTIPLICATION CHAINS ===
        SelfReflectionExample(
            input_question="A factory produces 15 toys per hour. It runs for 8 hours a day, 5 days a week. How many toys are produced in a week?",
            core_question="15 × 8 × 5 = ?",
            reasoning="Per hour: 15 toys\nPer day: 15 × 8 = 120 toys\nPer week: 120 × 5 = 600 toys",
            answer="600",
        ),
        SelfReflectionExample(
            input_question="A farmer has 4 fields. Each field has 6 rows. Each row has 10 plants. How many plants total?",
            core_question="4 × 6 × 10 = ?",
            reasoning="Rows total: 4 × 6 = 24 rows\nPlants total: 24 × 10 = 240 plants",
            answer="240",
        ),
        SelfReflectionExample(
            input_question="A hotel has 12 floors. Each floor has 8 rooms. Each room costs $150 per night. If all rooms are booked for one night, what is the total revenue?",
            core_question="12 × 8 × 150 = ?",
            reasoning="Total rooms: 12 × 8 = 96\nTotal revenue: 96 × 150 = $14,400",
            answer="$14,400",
        ),

        # === PERCENTAGE PROBLEMS ===
        SelfReflectionExample(
            input_question="A shirt costs $40. It's on sale for 25% off. What is the sale price?",
            core_question="40 - (40 × 0.25) = ?",
            reasoning="Discount: 40 × 0.25 = $10\nSale price: 40 - 10 = $30",
            answer="$30",
        ),
        SelfReflectionExample(
            input_question="A restaurant bill is $80. You want to leave a 15% tip. What is the total amount including tip?",
            core_question="80 + (80 × 0.15) = ?",
            reasoning="Tip: 80 × 0.15 = $12\nTotal: 80 + 12 = $92",
            answer="$92",
        ),
        SelfReflectionExample(
            input_question="A town had 2000 people. The population increased by 10%. Then it decreased by 5%. What is the current population?",
            core_question="2000 × 1.10 × 0.95 = ?",
            reasoning="After increase: 2000 × 1.10 = 2200\nAfter decrease: 2200 × 0.95 = 2090",
            answer="2090",
        ),
        SelfReflectionExample(
            input_question="A laptop costs $800. First, there's a 20% discount. Then, a 10% tax is applied to the discounted price. What's the final price?",
            core_question="(800 × 0.80) × 1.10 = ?",
            reasoning="After discount: 800 × 0.80 = $640\nAfter tax: 640 × 1.10 = $704",
            answer="$704",
        ),

        # === RATE PROBLEMS ===
        SelfReflectionExample(
            input_question="A car travels at 60 mph for 2 hours, then at 40 mph for 3 hours. What is the total distance traveled?",
            core_question="(60 × 2) + (40 × 3) = ?",
            reasoning="Distance at 60 mph: 60 × 2 = 120 miles\nDistance at 40 mph: 40 × 3 = 120 miles\nTotal: 120 + 120 = 240 miles",
            answer="240 miles",
        ),
        SelfReflectionExample(
            input_question="A pipe fills a tank at 5 gallons per minute. Another pipe drains it at 2 gallons per minute. If both are running, how many gallons are added in 10 minutes?",
            core_question="(5 - 2) × 10 = ?",
            reasoning="Net rate: 5 - 2 = 3 gallons/min\nIn 10 minutes: 3 × 10 = 30 gallons",
            answer="30 gallons",
        ),
        SelfReflectionExample(
            input_question="Worker A can complete a job in 6 hours. Worker B can complete it in 3 hours. If they work together, how long does it take?",
            core_question="1/(1/6 + 1/3) = ?",
            reasoning="A's rate: 1/6 job/hour\nB's rate: 1/3 = 2/6 job/hour\nCombined: 1/6 + 2/6 = 3/6 = 1/2 job/hour\nTime: 1 ÷ (1/2) = 2 hours",
            answer="2 hours",
        ),

        # === DISTRIBUTION/SHARING ===
        SelfReflectionExample(
            input_question="24 cookies are shared equally among 6 children. Each child eats 2 cookies. How many cookies does each child have left?",
            core_question="(24 ÷ 6) - 2 = ?",
            reasoning="Each child gets: 24 ÷ 6 = 4 cookies\nAfter eating 2: 4 - 2 = 2 cookies",
            answer="2",
        ),
        SelfReflectionExample(
            input_question="A pizza has 8 slices. Tom eats 1/4 of the pizza. How many slices are left?",
            core_question="8 - (8 × 1/4) = ?",
            reasoning="Tom eats: 8 × 1/4 = 2 slices\nRemaining: 8 - 2 = 6 slices",
            answer="6",
        ),
        SelfReflectionExample(
            input_question="Three friends split a $120 dinner bill. One friend pays for 2 others' drinks that cost $10 each. How much does that friend pay in total?",
            core_question="(120 ÷ 3) + (10 × 2) = ?",
            reasoning="Bill share: 120 ÷ 3 = $40\nDrinks paid: 10 × 2 = $20\nTotal: 40 + 20 = $60",
            answer="$60",
        ),

        # === COMPARISON PROBLEMS ===
        SelfReflectionExample(
            input_question="Alice has 3 times as many marbles as Bob. Bob has 2 more marbles than Carol. Carol has 5 marbles. How many marbles does Alice have?",
            core_question="3 × (5 + 2) = ?",
            reasoning="Carol: 5 marbles\nBob: 5 + 2 = 7 marbles\nAlice: 3 × 7 = 21 marbles",
            answer="21",
        ),
        SelfReflectionExample(
            input_question="A rope is 3 times as long as a stick. The stick is 4 feet long. If you cut 5 feet off the rope, how long is the remaining rope?",
            core_question="(3 × 4) - 5 = ?",
            reasoning="Rope length: 3 × 4 = 12 feet\nAfter cutting: 12 - 5 = 7 feet",
            answer="7 feet",
        ),
        SelfReflectionExample(
            input_question="Mike is twice as old as Tom. Tom is 5 years older than Sam. Sam is 10 years old. How old is Mike?",
            core_question="2 × (10 + 5) = ?",
            reasoning="Sam: 10 years\nTom: 10 + 5 = 15 years\nMike: 2 × 15 = 30 years",
            answer="30",
        ),

        # === MULTI-STEP WITH INTERMEDIATE VALUES ===
        SelfReflectionExample(
            input_question="A baker makes 50 cupcakes. He sells 30 at $2 each and gives away 10. He makes another batch of 25 cupcakes. How many cupcakes does he have now?",
            core_question="(50 - 30 - 10) + 25 = ?",
            reasoning="After selling: 50 - 30 = 20\nAfter giving: 20 - 10 = 10\nAfter new batch: 10 + 25 = 35",
            answer="35",
        ),
        SelfReflectionExample(
            input_question="A class has 30 students. 1/3 are absent today. Of those present, half join the field trip. How many students are on the field trip?",
            core_question="(30 - 30/3) × 0.5 = ?",
            reasoning="Absent: 30 ÷ 3 = 10\nPresent: 30 - 10 = 20\nOn field trip: 20 × 0.5 = 10",
            answer="10",
        ),
        SelfReflectionExample(
            input_question="A store buys 200 items at $5 each. They sell 150 items at $8 each. What is the total profit?",
            core_question="(150 × 8) - (200 × 5) = ?",
            reasoning="Cost: 200 × 5 = $1000\nRevenue: 150 × 8 = $1200\nProfit: 1200 - 1000 = $200",
            answer="$200",
        ),

        # === MONEY PROBLEMS ===
        SelfReflectionExample(
            input_question="John earns $15 per hour. He works 8 hours on Monday and 6 hours on Tuesday. He spends $50 on groceries. How much money does he have left?",
            core_question="15 × (8 + 6) - 50 = ?",
            reasoning="Total hours: 8 + 6 = 14\nEarnings: 15 × 14 = $210\nAfter groceries: 210 - 50 = $160",
            answer="$160",
        ),
        SelfReflectionExample(
            input_question="A book costs $12. A magazine costs $4. Sarah buys 3 books and 5 magazines. How much does she spend?",
            core_question="(12 × 3) + (4 × 5) = ?",
            reasoning="Books: 12 × 3 = $36\nMagazines: 4 × 5 = $20\nTotal: 36 + 20 = $56",
            answer="$56",
        ),
        SelfReflectionExample(
            input_question="Tom has $100. He buys a shirt for $25 and pants for $35. He then receives $15 as a gift. How much money does Tom have now?",
            core_question="100 - 25 - 35 + 15 = ?",
            reasoning="After shirt: 100 - 25 = $75\nAfter pants: 75 - 35 = $40\nAfter gift: 40 + 15 = $55",
            answer="$55",
        ),

        # === AGE PROBLEMS ===
        SelfReflectionExample(
            input_question="A mother is 30 years older than her daughter. In 5 years, the mother will be 45. How old is the daughter now?",
            core_question="(45 - 5) - 30 = ?",
            reasoning="Mother now: 45 - 5 = 40 years\nDaughter: 40 - 30 = 10 years",
            answer="10",
        ),
        SelfReflectionExample(
            input_question="The sum of two numbers is 50. One number is 14 more than the other. What is the smaller number?",
            core_question="x + (x + 14) = 50, find x",
            reasoning="Let smaller = x\nLarger = x + 14\nx + x + 14 = 50\n2x = 36\nx = 18",
            answer="18",
        ),

        # === FRACTIONS ===
        SelfReflectionExample(
            input_question="A tank is 1/4 full. After adding 30 gallons, it becomes 3/4 full. What is the tank's capacity?",
            core_question="30 gallons = 3/4 - 1/4 = 1/2 of capacity",
            reasoning="Added water fills: 3/4 - 1/4 = 1/2 of tank\n30 gallons = 1/2 capacity\nFull capacity: 30 × 2 = 60 gallons",
            answer="60 gallons",
        ),
        SelfReflectionExample(
            input_question="A rope is 24 meters long. 1/3 is cut off. Then 1/4 of the remaining rope is cut off. How long is the rope now?",
            core_question="24 × (2/3) × (3/4) = ?",
            reasoning="After first cut: 24 × 2/3 = 16 meters\nAfter second cut: 16 × 3/4 = 12 meters",
            answer="12 meters",
        ),

        # === TIME PROBLEMS ===
        SelfReflectionExample(
            input_question="A movie starts at 2:30 PM and lasts 2 hours 15 minutes. At what time does it end?",
            core_question="2:30 PM + 2h 15m = ?",
            reasoning="2:30 + 2 hours = 4:30\n4:30 + 15 minutes = 4:45 PM",
            answer="4:45 PM",
        ),
        SelfReflectionExample(
            input_question="A train leaves at 9:00 AM and arrives at 1:30 PM. How long is the journey in hours?",
            core_question="1:30 PM - 9:00 AM = ?",
            reasoning="9:00 AM to 1:00 PM = 4 hours\n1:00 PM to 1:30 PM = 30 minutes\nTotal: 4.5 hours",
            answer="4.5 hours",
        ),

        # === DISTANCE/SPEED/TIME ===
        SelfReflectionExample(
            input_question="A cyclist travels 45 km in 3 hours. At the same speed, how far will they travel in 5 hours?",
            core_question="(45/3) × 5 = ?",
            reasoning="Speed: 45 ÷ 3 = 15 km/h\nIn 5 hours: 15 × 5 = 75 km",
            answer="75 km",
        ),
        SelfReflectionExample(
            input_question="Two cars start 200 km apart and drive toward each other. One goes 50 km/h, the other 30 km/h. How long until they meet?",
            core_question="200 ÷ (50 + 30) = ?",
            reasoning="Combined speed: 50 + 30 = 80 km/h\nTime to meet: 200 ÷ 80 = 2.5 hours",
            answer="2.5 hours",
        ),

        # === AREA/PERIMETER ===
        SelfReflectionExample(
            input_question="A rectangle is 8 meters long and 5 meters wide. What is its perimeter?",
            core_question="2 × (8 + 5) = ?",
            reasoning="Perimeter = 2 × (length + width)\nPerimeter = 2 × (8 + 5) = 2 × 13 = 26 meters",
            answer="26 meters",
        ),
        SelfReflectionExample(
            input_question="A square garden has a perimeter of 36 meters. What is its area?",
            core_question="(36/4)² = ?",
            reasoning="Side = 36 ÷ 4 = 9 meters\nArea = 9 × 9 = 81 square meters",
            answer="81 square meters",
        ),

        # === AVERAGE PROBLEMS ===
        SelfReflectionExample(
            input_question="A student scored 85, 90, and 80 on three tests. What score is needed on the fourth test to have an average of 88?",
            core_question="(88 × 4) - (85 + 90 + 80) = ?",
            reasoning="Target total: 88 × 4 = 352\nCurrent total: 85 + 90 + 80 = 255\nNeeded: 352 - 255 = 97",
            answer="97",
        ),
        SelfReflectionExample(
            input_question="The average of 5 numbers is 20. If 4 of the numbers are 15, 18, 22, and 25, what is the fifth number?",
            core_question="(20 × 5) - (15 + 18 + 22 + 25) = ?",
            reasoning="Total: 20 × 5 = 100\nSum of 4 numbers: 15 + 18 + 22 + 25 = 80\nFifth number: 100 - 80 = 20",
            answer="20",
        ),

        # === UNIT CONVERSION ===
        SelfReflectionExample(
            input_question="A recipe needs 2.5 cups of flour. You have 10 tablespoons. There are 16 tablespoons in a cup. How many more tablespoons do you need?",
            core_question="(2.5 × 16) - 10 = ?",
            reasoning="Needed: 2.5 × 16 = 40 tablespoons\nHave: 10 tablespoons\nNeed more: 40 - 10 = 30 tablespoons",
            answer="30",
        ),
        SelfReflectionExample(
            input_question="A car uses 8 gallons of gas for 200 miles. How many gallons are needed for 350 miles?",
            core_question="8 × (350/200) = ?",
            reasoning="Miles per gallon: 200 ÷ 8 = 25 mpg\nGallons for 350: 350 ÷ 25 = 14 gallons",
            answer="14",
        ),

        # === PROFIT/LOSS ===
        SelfReflectionExample(
            input_question="A merchant buys goods for $500 and sells them for $650. What is the profit percentage?",
            core_question="(650 - 500) / 500 × 100 = ?",
            reasoning="Profit: 650 - 500 = $150\nPercentage: (150 / 500) × 100 = 30%",
            answer="30%",
        ),
        SelfReflectionExample(
            input_question="An item is marked up 40% from cost. If the selling price is $70, what was the cost?",
            core_question="70 / 1.4 = ?",
            reasoning="Selling price = Cost × 1.4\nCost = 70 ÷ 1.4 = $50",
            answer="$50",
        ),

        # === WORK PROBLEMS ===
        SelfReflectionExample(
            input_question="If 5 workers can build a wall in 10 days, how many days will it take 8 workers?",
            core_question="(5 × 10) / 8 = ?",
            reasoning="Total work: 5 × 10 = 50 worker-days\nWith 8 workers: 50 ÷ 8 = 6.25 days",
            answer="6.25 days",
        ),
        SelfReflectionExample(
            input_question="A job takes 12 hours with 4 people. If 2 more people join, how long will it take?",
            core_question="(4 × 12) / 6 = ?",
            reasoning="Total work: 4 × 12 = 48 person-hours\nWith 6 people: 48 ÷ 6 = 8 hours",
            answer="8 hours",
        ),

        # === COUNTING/COMBINATION ===
        SelfReflectionExample(
            input_question="A parking lot has 5 rows with 12 spaces each. 17 spaces are occupied. How many spaces are empty?",
            core_question="(5 × 12) - 17 = ?",
            reasoning="Total spaces: 5 × 12 = 60\nEmpty: 60 - 17 = 43",
            answer="43",
        ),
        SelfReflectionExample(
            input_question="A box contains 3 red, 5 blue, and 7 green marbles. If you remove 4 blue marbles, what fraction of the remaining marbles are green?",
            core_question="7 / (3 + 1 + 7) = ?",
            reasoning="After removal: 3 red, 1 blue, 7 green = 11 total\nGreen fraction: 7/11",
            answer="7/11",
        ),

        # === COMPLEX MULTI-STEP ===
        SelfReflectionExample(
            input_question="A car rental costs $30 per day plus $0.25 per mile. John rents for 3 days and drives 200 miles. What is the total cost?",
            core_question="(30 × 3) + (0.25 × 200) = ?",
            reasoning="Daily charge: 30 × 3 = $90\nMileage: 0.25 × 200 = $50\nTotal: 90 + 50 = $140",
            answer="$140",
        ),
        SelfReflectionExample(
            input_question="A phone plan costs $50 per month plus $0.10 per text over 500. If you send 750 texts, what's the monthly bill?",
            core_question="50 + (0.10 × (750 - 500)) = ?",
            reasoning="Extra texts: 750 - 500 = 250\nExtra cost: 0.10 × 250 = $25\nTotal: 50 + 25 = $75",
            answer="$75",
        ),
        SelfReflectionExample(
            input_question="A store offers buy 2 get 1 free on $5 items. If you want 7 items, how much do you pay?",
            core_question="How many free items in 7, then calculate",
            reasoning="7 items = 2 groups of 3 (2 free) + 1 extra\nPay for: 7 - 2 = 5 items\nCost: 5 × 5 = $25",
            answer="$25",
        ),
        SelfReflectionExample(
            input_question="A pool is filled by pipe A in 6 hours and drained by pipe B in 9 hours. Starting empty with both open, how long to fill?",
            core_question="1 / (1/6 - 1/9) = ?",
            reasoning="Fill rate: 1/6 per hour\nDrain rate: 1/9 per hour\nNet rate: 1/6 - 1/9 = 3/18 - 2/18 = 1/18 per hour\nTime: 18 hours",
            answer="18 hours",
        ),
    ]
