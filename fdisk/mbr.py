import io
import struct
from dataclasses import dataclass
from typing import Optional, List, Union
from enum import IntEnum


# From http://www.osdever.net/documents/partitiontypes.php
class PartitionId(IntEnum):
    EMPTY = 0x00
    FAT12 = 0x01
    XENIX_ROOT = 0x02
    XENIX_USR = 0x03
    DOS_FAT16_LT_32M = 0x04
    DOS_EXTENDED = 0x05
    DOS_FAT16 = 0x06
    NT_NTFS = 0x07
    EX_FAT = 0x07
    HPFS = 0x07
    QNX2 = 0x07
    OS2_LT_V13 = 0x08
    OS2_BOOT_MANAGER = 0x0a
    WIN95_FAT32 = 0x0b
    WIN95_FAT32_LBA = 0x0c
    WIN95_FAT16_LBA = 0x0e
    WIN95_EXT_LBA = 0x0f
    HIDDEN_FAT12 = 0x11
    HIDDEN_DOS_FAT16_LT_32M = 0x14
    HIDDEN_DOS_FAT16 = 0x16
    HIDDEN_HPFS = 0x17
    HIDDEN_WIN95_FAT32 = 0x1b
    HIDDEN_WIN95_FAT32_LBA = 0x1c
    HIDDEN_WIN95_FAT16_LBA = 0x1e
    PLAN_9 = 0x39
    LINUX_SWAP = 0x82
    LINUX = 0x83
    LINUX_EXTENDED = 0x85
    LINUX_LLVM = 0x8e
    HIDDEN_LINUX = 0x93
    FREEBSD = 0xa5
    OPENBSD = 0xa6
    NETBSD = 0xa9
    HFS_HFS_PLUS = 0xaf
    GPT = 0xee
    EFI = 0xef
    LINUX_RAID = 0xfd


SECTOR_SIZE = 512


@dataclass()
class LogicalBlockAddress:
    """Logical block addressing"""
    sector: int

    def byte_offset(self) -> int:
        return self.sector * SECTOR_SIZE

    def __bytes__(self) -> bytes:
        return struct.pack("<I", self.sector)


@dataclass()
class CHSAddress:
    """Legacy Cylinder Head Sector addressing"""
    cylinder: int  # 0 .. 1024
    head: int  # 0 .. 255
    sector: int  # 0 .. 63

    @staticmethod
    def parse(b: bytes) -> 'CHSAddress':
        head = b[0]
        sector = b[1] & 0x3F
        cylinder = b[2] | ((b[1] >> 6) << 8)
        return CHSAddress(cylinder, head, sector)

    def __bytes__(self) -> bytes:
        assert 0 <= self.cylinder <= 1024
        assert 0 <= self.head <= 255
        assert 0 <= self.sector <= 63
        b1 = self.head  # 8 bits of head
        b2 = self.sector | (self.cylinder >> 8) << 6  # 6 bits + upper 2 bits of cylinder
        b3 = self.cylinder & 0xFF  # lower 8 bits of cylinder
        return bytes([b1, b2, b3])


PARTITION_ENTRY_SIZE = 16


@dataclass()
class PartitionEntry:
    bootable: bool
    partition_type: Union[int, PartitionId]
    start_block_address: LogicalBlockAddress
    sector_count: int
    start_chs: Optional[CHSAddress]  # legacy
    end_chs: Optional[CHSAddress]  # legacy

    @property
    def end_block_address(self) -> LogicalBlockAddress:
        if self.sector_count == 0:
            return self.start_block_address
        return LogicalBlockAddress(self.start_block_address.sector + self.sector_count - 1)

    @property
    def byte_count(self) -> int:
        return self.sector_count * SECTOR_SIZE

    @staticmethod
    def read(fp: Union[io.BufferedIOBase, bytes, bytearray]) -> Optional['PartitionEntry']:
        """Read from bytes, bytearray, or binary file stream"""
        bootable_partition_flag = 0x80
        raw = None
        if isinstance(fp, (bytes, bytearray)):
            raw = fp
        elif isinstance(fp, io.BufferedIOBase):
            raw = fp.read(PARTITION_ENTRY_SIZE)
        else:
            raise ValueError("fp - must be bytes or fp")

        if len(raw) < PARTITION_ENTRY_SIZE:
            return None

        # Partition Entry Format
        # | 0x00 | 1 | Drive attributes (bit 7 set = active or bootable) |
        # | 0x01 | 3 | CHS Address of partition start                    |
        # | 0x04 | 1 | Partition type                                    |
        # | 0x05 | 3 | CHS address of last partition sector              |
        # | 0x08 | 4 | LBA of partition start                            |
        # | 0x0C | 4 | Number of sectors in partition                    |

        attrib, part_type, lba_start, sector_count = struct.unpack("<BxxxBxxxII", raw)
        if part_type in PartitionId.__members__.values():
            part_type = PartitionId(part_type)

        start_chs = CHSAddress.parse(raw[1:4])
        end_chs = CHSAddress.parse(raw[5:8])
        return PartitionEntry((attrib & bootable_partition_flag) != 0,
                              part_type,
                              LogicalBlockAddress(lba_start),
                              sector_count, start_chs, end_chs)

    def __bytes__(self) -> bytes:
        """Convert to bytes"""
        # Partition Entry Format
        # | 0x00 | 1 | Drive attributes (bit 7 set = active or bootable) |
        # | 0x01 | 3 | CHS Address of partition start                    |
        # | 0x04 | 1 | Partition type                                    |
        # | 0x05 | 3 | CHS address of last partition sector              |
        # | 0x08 | 4 | LBA of partition start                            |
        # | 0x0C | 4 | Number of sectors in partition                    |
        return bytes([0x80 if self.bootable else 0x00]) + \
               bytes(self.start_chs) + \
               bytes([int(self.partition_type)]) + \
               bytes(self.end_chs) + \
               bytes(self.start_block_address) + \
               struct.pack("<I", self.sector_count)


UNIQUE_ID_OFFSET = 0x1B8
UNIQUE_ID_SIZE = 4
PARTITION_OFFSETS = [0x1BE, 0x1CE, 0x1DE, 0x1EE]
VALID_BOOT_SECTOR_OFFSET = 0x1FE
VALID_BOOT_SECTOR_SIZE = 2
VALID_MARKER = 0xaa55
MBR_SIZE = 512


@dataclass()
class MasterBootRecord:
    unique_id: Optional[int]
    partitions: List[Optional[PartitionEntry]]

    @staticmethod
    def read(fp: Union[bytes, bytearray, io.BufferedIOBase], ignore_valid_marker=False) -> Optional['MasterBootRecord']:
        """Read master boot record from bytes, bytearray, or binary file stream"""
        if isinstance(fp, (bytes, bytearray)):
            data = fp
        elif isinstance(fp, io.BufferedIOBase):
            data = fp.read(MBR_SIZE)
        else:
            raise ValueError("fp - must be bytes or fp")
        # MBR Format
        # | 0x1B8 | 4 | Optional "Unique Disk ID / Signature"            |
        # | 0x1BC | 2  | Optional, reserved 0x00003                      |
        # | 0x1BE | 16 | First partition table entry                     |
        # | 0x1CE | 16 | Second partition table entry                    |
        # | 0x1DE | 16 | Third partition table entry                     |
        # | 0x1EE | 16 | Fourth partition table entry                    |
        # | 0x1FE | 2  | (0x55, 0xAA) "Valid bootsector" signature bytes |
        unique_id, = struct.unpack("<I", data[UNIQUE_ID_OFFSET: UNIQUE_ID_OFFSET + UNIQUE_ID_SIZE])
        partitions = [PartitionEntry.read(data[offset: offset + PARTITION_ENTRY_SIZE]) for offset in PARTITION_OFFSETS]
        valid_marker, = struct.unpack("<H", data[VALID_BOOT_SECTOR_OFFSET:
                                                 VALID_BOOT_SECTOR_OFFSET + VALID_BOOT_SECTOR_SIZE])
        if ignore_valid_marker or valid_marker == VALID_MARKER:
            return MasterBootRecord(unique_id, partitions)
        return None

    def __bytes__(self) -> bytes:
        """Convert to bytes"""
        wr = io.BytesIO(bytes([0] * 512))
        wr.seek(UNIQUE_ID_OFFSET, io.SEEK_SET)
        wr.write(struct.pack("<I", self.unique_id or 0))
        for i, partition in enumerate(self.partitions):
            offset = PARTITION_OFFSETS[i]
            wr.seek(offset, io.SEEK_SET)
            wr.write(bytes(partition))
        wr.seek(VALID_BOOT_SECTOR_OFFSET, io.SEEK_SET)
        wr.write(struct.pack("<H", VALID_MARKER))
        return wr.getvalue()
