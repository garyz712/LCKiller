from abc import ABC, abstractmethod
import sys

class InMemoryDB(ABC):
    @abstractmethod
    def set(self, timestamp: int, key: str, field: str, value: int) -> None:
        """Set a value for a key-field pair at a given timestamp."""
        pass

    @abstractmethod
    def set_with_ttl(self, timestamp: int, key: str, field: str, value: int, ttl: int) -> None:
        """Set a value for a key-field pair with a time-to-live (TTL)."""
        pass

    @abstractmethod
    def compare_and_set(self, timestamp: int, key: str, field: str, expected_value: int, new_value: int) -> bool:
        """Atomically set a value only if the current value matches the expected value."""
        pass

    @abstractmethod
    def compare_and_set_with_ttl(self, timestamp: int, key: str, field: str, expected_value: int, new_value: int, ttl: int) -> bool:
        """Atomically set a value with TTL only if the current value matches the expected value."""
        pass

    @abstractmethod
    def compare_and_delete(self, timestamp: int, key: str, field: str, expected_value: int) -> bool:
        """Atomically delete a field only if the current value matches the expected value."""
        pass

    @abstractmethod
    def get(self, timestamp: int, key: str, field: str) -> int | None:
        """Get the value of a key-field pair at a given timestamp."""
        pass

    @abstractmethod
    def get_when(self, timestamp: int, key: str, field: str, at_timestamp: int) -> int | None:
        """Get the value of a key-field pair at a specific historical timestamp."""
        pass

    @abstractmethod
    def scan(self, timestamp: int, key: str) -> list[str]:
        """Scan all fields for a given key at a given timestamp."""
        pass

    @abstractmethod
    def scan_by_prefix(self, timestamp: int, key: str, prefix: str) -> list[str]:
        """Scan fields for a given key that match a prefix at a given timestamp."""
        pass


class InMemoryDBImpl(InMemoryDB):
    def __init__(self):
        self.db = {}

    # find the latest value of db[key][field] history at query_ts, if expired, return None; if deleted, return val=None; if not expired, return the value   
    def _get_value_at(self, key: str, field: str, query_ts: int) -> int | None:
        if key not in self.db or field not in self.db[key]:
            return None
        history = self.db[key][field]
        for i in range(len(history) - 1, -1, -1):
            set_ts, val, exp = history[i]

            if exp and query_ts >= exp:  #if the latest set has expired date and it is expired at query_ts, return None
                return None

            if set_ts > query_ts: # if there is no expired date or it is not expired, check if the set_ts is smaller than query_ts
                continue
            
            # if the set_ts is smaller than query_ts and it is not expired, return the latest valid value, which can be None if deleted
            return val 
        return None

    def set(self, timestamp: int, key: str, field: str, value: int) -> None:
        if key not in self.db:
            self.db[key] = {}
        if field not in self.db[key]:
            self.db[key][field] = []
        self.db[key][field].append((timestamp, value, None))

    def set_with_ttl(self, timestamp: int, key: str, field: str, value: int, ttl: int) -> None:
        if key not in self.db:
            self.db[key] = {}
        if field not in self.db[key]:
            self.db[key][field] = []
        exp = timestamp + ttl 
        self.db[key][field].append((timestamp, value, exp))

    def compare_and_set(self, timestamp: int, key: str, field: str, expected_value: int, new_value: int) -> bool:
        current = self._get_value_at(key, field, timestamp)
        if current == expected_value:
            if key not in self.db:
                self.db[key] = {}
            if field not in self.db[key]:
                self.db[key][field] = []
            self.db[key][field].append((timestamp, new_value, None))
            return True
        return False

    def compare_and_set_with_ttl(self, timestamp: int, key: str, field: str, expected_value: int, new_value: int, ttl: int) -> bool:
        current = self._get_value_at(key, field, timestamp)
        if current == expected_value:
            if key not in self.db:
                self.db[key] = {}
            if field not in self.db[key]:
                self.db[key][field] = []
            exp = timestamp + ttl 
            self.db[key][field].append((timestamp, new_value, exp))
            return True
        return False

    def compare_and_delete(self, timestamp: int, key: str, field: str, expected_value: int) -> bool:
        current = self._get_value_at(key, field, timestamp)
        if current == expected_value:
            if key not in self.db:
                self.db[key] = {}
            if field not in self.db[key]:
                self.db[key][field] = []
            self.db[key][field].append((timestamp, None, None))
            return True
        return False

    def get(self, timestamp: int, key: str, field: str) -> int | None:
        return self._get_value_at(key, field, timestamp)

    def get_when(self, timestamp: int, key: str, field: str, at_timestamp: int) -> int | None:
        if at_timestamp == 0:
            return self.get(timestamp, key, field)
        return self._get_value_at(key, field, at_timestamp)

    def scan(self, timestamp: int, key: str) -> list[str]:
        if key not in self.db:
            return []
        result = []
        for field, history in self.db[key].items():
            val = self._get_value_at(key, field, timestamp)
            if val is not None:
                result.append(f"{field}({val})")
        result.sort()
        return result

    def scan_by_prefix(self, timestamp: int, key: str, prefix: str) -> list[str]:
        if key not in self.db:
            return []
        result = []
        for field, history in self.db[key].items():
            if field.startswith(prefix):
                val = self._get_value_at(key, field, timestamp)
                if val is not None:
                    result.append(f"{field}({val})")
        result.sort()
        return result


# tests/level_2_tests.py  (or wherever your Level 2 tests are)

import unittest

class TestLevel2Case10(unittest.TestCase):
    def test_level_2_case_10_mixed_multiple_operations_2(self):
        db = InMemoryDBImpl()
        fixed_timestamp = 0  # We'll increment manually

        # set(0, "a", "a", 1)
        db.set(fixed_timestamp, "a", "a", 1)
        fixed_timestamp += 1

        # set(1, "a", "b", 0)
        db.set(fixed_timestamp, "a", "b", 0)
        fixed_timestamp += 1

        # scan(2, "a") → should return ["a(1)", "b(0)"]
        expected = ["a(1)", "b(0)"]
        self.assertEqual(db.scan(fixed_timestamp, "a"), expected)

        #print(db.scan(fixed_timestamp, "a"))
        fixed_timestamp += 1

        # compare_and_delete(3, "a", "a", 1)
        self.assertTrue(db.compare_and_delete(fixed_timestamp, "a", "a", 1))
        fixed_timestamp += 1

        # compare_and_delete(4, "a", "b", 0)
        self.assertTrue(db.compare_and_delete(fixed_timestamp, "a", "b", 0))
        fixed_timestamp += 1

        # compare_and_set(5, "a", "b", 0, 7) → should fail (field doesn't exist)
        self.assertFalse(db.compare_and_set(fixed_timestamp, "a", "b", 0, 7))
        fixed_timestamp += 1

        # set(6, "a", "b", 12)
        db.set(fixed_timestamp, "a", "b", 12)
        fixed_timestamp += 1

        # scan(7, "a") → should return ["b(12)"]
        expected = ["b(12)"]
        self.assertEqual(db.scan(fixed_timestamp, "a"), expected)
        fixed_timestamp += 1

        # scan(8, "a") → same
        self.assertEqual(db.scan(fixed_timestamp, "a"), expected)
        fixed_timestamp += 1

        # get(9, "a", "b") → 12
        self.assertEqual(db.get(fixed_timestamp, "a", "b"), 12)
        fixed_timestamp += 1

        # set(10, "b", "b", 18)
        db.set(fixed_timestamp, "b", "b", 18)
        fixed_timestamp += 1

        # scan(11, "a") → still ["b(12)"]
        self.assertEqual(db.scan(fixed_timestamp, "a"), ["b(12)"])

        # (Optional) You can add more if you want to test key "b" too
        # self.assertEqual(db.scan(fixed_timestamp, "b"), ["b(18)"])



class TestInMemoryDB(unittest.TestCase):

    # --------------------------------------------------------------
    # Helper to create DB with incremental timestamps
    # --------------------------------------------------------------
    def _db(self):
        db = InMemoryDBImpl()
        db.ts = 0
        def inc():
            db.ts += 1
            return db.ts
        db.next_ts = inc
        return db

    # --------------------------------------------------------------
    # LEVEL 1: Basic set/get/compare_and_set/delete
    # --------------------------------------------------------------
    def test_level1_basic(self):
        db = self._db()
        ts = db.next_ts

        db.set(ts(), "user1", "score", 100)
        self.assertEqual(db.get(ts(), "user1", "score"), 100)

        self.assertTrue(db.compare_and_set(ts(), "user1", "score", 100, 200))
        self.assertEqual(db.get(ts(), "user1", "score"), 200)

        self.assertFalse(db.compare_and_set(ts(), "user1", "score", 100, 300))
        self.assertEqual(db.get(ts(), "user1", "score"), 200)

        self.assertTrue(db.compare_and_delete(ts(), "user1", "score", 200))
        self.assertIsNone(db.get(ts(), "user1", "score"))

    # --------------------------------------------------------------
    # LEVEL 2: scan and scan_by_prefix
    # --------------------------------------------------------------
    def test_level2_scan(self):
        db = self._db()
        ts = db.next_ts

        db.set(ts(), "A", "age", 25)
        db.set(ts(), "A", "name", 1)
        db.set(ts(), "A", "city", 5)
        db.set(ts(), "B", "x", 99)

        self.assertEqual(
            db.scan(ts(), "A"),
            ["age(25)", "city(5)", "name(1)"]
        )
        self.assertEqual(db.scan(ts(), "B"), ["x(99)"])
        self.assertEqual(db.scan(ts(), "C"), [])

    def test_level2_scan_by_prefix(self):
        db = self._db()
        ts = db.next_ts

        db.set(ts(), "X", "aa", 1)
        db.set(ts(), "X", "ab", 2)
        db.set(ts(), "X", "ba", 3)
        db.set(ts(), "X", "bb", 4)

        self.assertEqual(
            db.scan_by_prefix(ts(), "X", "a"),
            ["aa(1)", "ab(2)"]
        )
        self.assertEqual(
            db.scan_by_prefix(ts(), "X", "b"),
            ["ba(3)", "bb(4)"]
        )
        self.assertEqual(db.scan_by_prefix(ts(), "X", "z"), [])
        self.assertEqual(db.scan_by_prefix(ts(), "Y", "a"), [])

    def test_level2_mixed_operations_failing_case(self):
        # This is the exact test from your screenshot
        db = self._db()
        ts = db.next_ts

        db.set(ts(), "a", "a", 1)
        db.set(ts(), "a", "b", 0)

        self.assertEqual(
            db.scan(ts(), "a"),
            ["a(1)", "b(0)"]
        )

        self.assertTrue(db.compare_and_delete(ts(), "a", "a", 1))
        self.assertTrue(db.compare_and_delete(ts(), "a", "b", 0))
        self.assertFalse(db.compare_and_set(ts(), "a", "b", 0, 7))

        db.set(ts(), "a", "b", 12)
        self.assertEqual(db.scan(ts(), "a"), ["b(12)"])
        self.assertEqual(db.get(ts(), "a", "b"), 12)

        db.set(ts(), "b", "b", 18)
        self.assertEqual(db.scan(ts(), "a"), ["b(12)"])

    # --------------------------------------------------------------
    # LEVEL 3: TTL support
    # --------------------------------------------------------------
    def test_level3_ttl_expiration(self):
        db = self._db()
        ts = db.next_ts


        db.set_with_ttl(ts(), "A", "x", 10, 5)   # expires at ts=5
        db.set_with_ttl(ts(), "A", "y", 20, 10)  # expires at ts=10

        self.assertEqual(db.get(ts(), "A", "x"), 10)
        self.assertEqual(db.scan(ts(), "A"), ["x(10)", "y(20)"])

        # Advance time
        db.ts = 6
        result = db.get(db.ts, "A", "x")
        debug_msg = f"[DEBUG] testing... db.get({db.ts}, 'A', 'x') = {result}"
        print(debug_msg, file=sys.stderr, flush=True)
        print(debug_msg)  # Also print to stdout in case IDE only shows stdout
        self.assertIsNone(result)
        self.assertEqual(db.scan(db.ts, "A"), ["y(20)"])

        db.ts = 12
        
        self.assertIsNone(db.get(db.ts, "A", "y"))
        self.assertEqual(db.scan(db.ts, "A"), [])

    def test_level3_compare_and_set_with_ttl(self):
        db = self._db()
        ts = db.next_ts

        db.set(ts(), "A", "f", 5)
        self.assertTrue(db.compare_and_set_with_ttl(ts(), "A", "f", 5, 10, 3))

        self.assertEqual(db.get(ts(), "A", "f"), 10)
        db.ts = ts() + 4

        print("testing... !!!!!!", db.db)
        self.assertIsNone(db.get(db.ts, "A", "f"))

    # --------------------------------------------------------------
    # LEVEL 4: get_when (look-back)
    # --------------------------------------------------------------
    def test_level4_get_when(self):
        db = self._db()
        ts = db.next_ts

        db.set_with_ttl(ts(), "A", "score", 100, 10)  # expires at 10
        db.compare_and_set_with_ttl(ts(), "A", "score", 100, 200, 5)

        self.assertEqual(db.get_when(ts(), "A", "score", 0), 200)  # current
        self.assertEqual(db.get_when(ts(), "A", "score", 1), 100)  # original
        self.assertEqual(db.get_when(ts(), "A", "score", 5), 200)
        self.assertIsNone(db.get_when(ts(), "A", "score", 11))   # expired

    def test_level4_get_when_nonexistent(self):
        db = self._db()
        ts = db.next_ts

        self.assertIsNone(db.get_when(ts(), "X", "f", 0))
        self.assertIsNone(db.get_when(ts(), "X", "f", 100))

    # --------------------------------------------------------------
    # Edge Cases & Stress
    # --------------------------------------------------------------
    def test_edge_empty_db(self):
        db = InMemoryDBImpl()
        self.assertEqual(db.scan(0, "any"), [])
        self.assertIsNone(db.get(0, "any", "any"))

    def test_edge_ttl_zero(self):
        db = self._db()
        ts = db.next_ts

        db.set_with_ttl(ts(), "A", "x", 1, 0)   # ttl=0 → infinite
        db.ts = 1000
        self.assertIsNone(db.get(db.ts, "A", "x"))



    def test_edge_field_reuse_after_delete(self):
        db = self._db()
        ts = db.next_ts

        db.set(ts(), "K", "f", 1)
        self.assertTrue(db.compare_and_delete(ts(), "K", "f", 1))
        db.set(ts(), "K", "f", 2)
        self.assertEqual(db.get(ts(), "K", "f"), 2)

    def test_stress_many_fields(self):
        db = self._db()
        ts = db.next_ts

        for i in range(100):
            db.set(ts(), "K", f"field_{i}", i)

        result = db.scan(ts(), "K")
        self.assertEqual(len(result), 100)
        self.assertIn("field_42(42)", result)

        prefix_result = db.scan_by_prefix(ts(), "K", "field_5")
        self.assertTrue(all(f.startswith("field_5") for f in [x.split("(")[0] for x in prefix_result]))

    def test_compare_and_set_on_expired_field(self):
        db = self._db()
        ts = db.next_ts

        db.set_with_ttl(ts(), "A", "x", 1, 2)
        db.ts = 5  # expired
        self.assertFalse(db.compare_and_set(db.ts, "A", "x", 1, 99))
        self.assertIsNone(db.get(db.ts, "A", "x"))


if __name__ == '__main__':
    # Disable buffering to see print statements immediately
    # Debug output is written to stderr, so it appears immediately even with unittest buffering
    unittest.main(verbosity=2, buffer=False)



