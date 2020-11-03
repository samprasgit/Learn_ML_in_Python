class Solution1:
    def intersectoion(self, nums1, nums2):
        # set
        set1 = set(nums1)
        set2 = set(nums2)
        return self.set_intersection(set1, set2)

    def set_intersection(self, set1, set2):
        if len(set1) > len(set2):
            return self.set_intersection(set2, set1)
        return [x for x in set1 if x in set2]
