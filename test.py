# https://huggingface.co/collections/livebench/livebench-67eaef9bb68b45b17a197a98
from datasets import Dataset, load_dataset, load_from_disk

ds = load_from_disk("livebench_coding")
df = ds['test'].to_pandas()
"""
Index(['question_id', 'category', 'turns', 'question_title',
       'public_test_cases', 'private_test_cases', 'original_json',
       'release_date', 'citation', 'task', 'livebench_release_date',
       'livebench_removal_date', 'remainder', 'solution', 'partial_solution'],
      dtype='object')
"""

# Get agentic coding livebench 

for didx, d in df.iterrows():
    print(' '.join(d['turns']))

    assert(False)
    """
    ChatGPT-5 Answer: 

    from typing import List
    from math import gcd
    from collections import Counter

    class Solution:
        def minimumArrayLength(self, nums: List[int]) -> int:
            # Find the minimum element
            mn = min(nums)
            # Compute gcd of all numbers
            g = 0
            for x in nums:
                g = gcd(g, x)
            # If the gcd is not equal to the minimum, we cannot reduce to a single number
            if g != mn:
                return 1
            
            # Count how many times the minimum appears
            c = nums.count(mn)
            # We can pair up the minimum numbers with each other to remove them two at a time.
            # If we have c such numbers, we can reduce them until at most ceil(c/2) remain.
            # The minimum length is ceil(c/2).
            return (c + 1) // 2
    """