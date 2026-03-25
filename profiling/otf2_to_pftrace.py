from os import PathLike
import otf2
from otf2.events import *
import perfetto
import sys


def read_otf2(path: PathLike):
    with otf2.reader.open(path) as trace:
        events = dict()
        events["AttributeList"] = 0
        events["Base"] = 0
        events["BufferFlush"] = 0
        events["CallingContextEnter"] = 0
        events["CallingContextLeave"] = 0
        events["CallingContextSample"] = 0
        events["CommCreate"] = 0
        events["CommDestroy"] = 0
        events["Enter"] = 0
        events["Integral"] = 0
        events["IoAcquireLock"] = 0
        events["IoChangeStatusFlags"] = 0
        events["IoCreateHandle"] = 0
        events["IoDeleteFile"] = 0
        events["IoDestroyHandle"] = 0
        events["IoDuplicateHandle"] = 0
        events["IoOperationBegin"] = 0
        events["IoOperationCancelled"] = 0
        events["IoOperationComplete"] = 0
        events["IoOperationIssued"] = 0
        events["IoOperationTest"] = 0
        events["IoReleaseLock"] = 0
        events["IoSeek"] = 0
        events["IoTryLock"] = 0
        events["Leave"] = 0
        events["MeasurementOnOff"] = 0
        events["Metric"] = 0
        events["MpiCollectiveBegin"] = 0
        events["MpiCollectiveEnd"] = 0
        events["MpiIrecv"] = 0
        events["MpiIrecvRequest"] = 0
        events["MpiIsend"] = 0
        events["MpiIsendComplete"] = 0
        events["MpiRecv"] = 0
        events["MpiRequestCancelled"] = 0
        events["MpiRequestTest"] = 0
        events["MpiSend"] = 0
        events["NonBlockingCollectiveComplete"] = 0
        events["NonBlockingCollectiveRequest"] = 0
        events["Number"] = 0
        events["OmpAcquireLock"] = 0
        events["OmpFork"] = 0
        events["OmpJoin"] = 0
        events["OmpReleaseLock"] = 0
        events["OmpTaskComplete"] = 0
        events["OmpTaskCreate"] = 0
        events["OmpTaskSwitch"] = 0
        events["ParameterInt"] = 0
        events["ParameterString"] = 0
        events["ParameterType"] = 0
        events["ParameterUnsignedInt"] = 0
        events["ProgramBegin"] = 0
        events["ProgramEnd"] = 0
        events["RmaAcquireLock"] = 0
        events["RmaAtomic"] = 0
        events["RmaCollectiveBegin"] = 0
        events["RmaCollectiveEnd"] = 0
        events["RmaGet"] = 0
        events["RmaGroupSync"] = 0
        events["RmaOpCompleteBlocking"] = 0
        events["RmaOpCompleteNonBlocking"] = 0
        events["RmaOpCompleteRemote"] = 0
        events["RmaOpTest"] = 0
        events["RmaPut"] = 0
        events["RmaReleaseLock"] = 0
        events["RmaRequestLock"] = 0
        events["RmaSync"] = 0
        events["RmaTryLock"] = 0
        events["RmaWaitChange"] = 0
        events["RmaWinCreate"] = 0
        events["RmaWinDestroy"] = 0
        events["ThreadAcquireLock"] = 0
        events["ThreadBegin"] = 0
        events["ThreadCreate"] = 0
        events["ThreadEnd"] = 0
        events["ThreadFork"] = 0
        events["ThreadJoin"] = 0
        events["ThreadReleaseLock"] = 0
        events["ThreadTaskComplete"] = 0
        events["ThreadTaskCreate"] = 0
        events["ThreadTaskSwitch"] = 0
        events["ThreadTeamBegin"] = 0
        events["ThreadTeamEnd"] = 0
        events["ThreadWait"] = 0
        events["Unnamed"] = 0

        for _, event in trace.events:
            if isinstance(event, AttributeList):
                events["AttributeList"] += 1
            elif isinstance(event, Base):
                events["Base"] += 1
            elif isinstance(event, BufferFlush):
                events["BufferFlush"] += 1
            elif isinstance(event, CallingContextEnter):
                events["CallingContextEnter"] += 1
            elif isinstance(event, CallingContextLeave):
                events["CallingContextLeave"] += 1
            elif isinstance(event, CallingContextSample):
                events["CallingContextSample"] += 1
            elif isinstance(event, CommCreate):
                events["CommCreate"] += 1
            elif isinstance(event, CommDestroy):
                events["CommDestroy"] += 1
            elif isinstance(event, Enter):
                events["Enter"] += 1
            elif isinstance(event, Integral):
                events["Integral"] += 1
            elif isinstance(event, IoAcquireLock):
                events["IoAcquireLock"] += 1
            elif isinstance(event, IoChangeStatusFlags):
                events["IoChangeStatusFlags"] += 1
            elif isinstance(event, IoCreateHandle):
                events["IoCreateHandle"] += 1
            elif isinstance(event, IoDeleteFile):
                events["IoDeleteFile"] += 1
            elif isinstance(event, IoDestroyHandle):
                events["IoDestroyHandle"] += 1
            elif isinstance(event, IoDuplicateHandle):
                events["IoDuplicateHandle"] += 1
            elif isinstance(event, IoOperationBegin):
                events["IoOperationBegin"] += 1
            elif isinstance(event, IoOperationCancelled):
                events["IoOperationCancelled"] += 1
            elif isinstance(event, IoOperationComplete):
                events["IoOperationComplete"] += 1
            elif isinstance(event, IoOperationIssued):
                events["IoOperationIssued"] += 1
            elif isinstance(event, IoOperationTest):
                events["IoOperationTest"] += 1
            elif isinstance(event, IoReleaseLock):
                events["IoReleaseLock"] += 1
            elif isinstance(event, IoSeek):
                events["IoSeek"] += 1
            elif isinstance(event, IoTryLock):
                events["IoTryLock"] += 1
            elif isinstance(event, Leave):
                events["Leave"] += 1
            elif isinstance(event, MeasurementOnOff):
                events["MeasurementOnOff"] += 1
            elif isinstance(event, Metric):
                events["Metric"] += 1
            elif isinstance(event, MpiCollectiveBegin):
                events["MpiCollectiveBegin"] += 1
            elif isinstance(event, MpiCollectiveEnd):
                events["MpiCollectiveEnd"] += 1
            elif isinstance(event, MpiIrecv):
                events["MpiIrecv"] += 1
            elif isinstance(event, MpiIrecvRequest):
                events["MpiIrecvRequest"] += 1
            elif isinstance(event, MpiIsend):
                events["MpiIsend"] += 1
            elif isinstance(event, MpiIsendComplete):
                events["MpiIsendComplete"] += 1
            elif isinstance(event, MpiRecv):
                events["MpiRecv"] += 1
            elif isinstance(event, MpiRequestCancelled):
                events["MpiRequestCancelled"] += 1
            elif isinstance(event, MpiRequestTest):
                events["MpiRequestTest"] += 1
            elif isinstance(event, MpiSend):
                events["MpiSend"] += 1
            elif isinstance(event, NonBlockingCollectiveComplete):
                events["NonBlockingCollectiveComplete"] += 1
            elif isinstance(event, NonBlockingCollectiveRequest):
                events["NonBlockingCollectiveRequest"] += 1
            elif isinstance(event, Number):
                events["Number"] += 1
            elif isinstance(event, OmpAcquireLock):
                events["OmpAcquireLock"] += 1
            elif isinstance(event, OmpFork):
                events["OmpFork"] += 1
            elif isinstance(event, OmpJoin):
                events["OmpJoin"] += 1
            elif isinstance(event, OmpReleaseLock):
                events["OmpReleaseLock"] += 1
            elif isinstance(event, OmpTaskComplete):
                events["OmpTaskComplete"] += 1
            elif isinstance(event, OmpTaskCreate):
                events["OmpTaskCreate"] += 1
            elif isinstance(event, OmpTaskSwitch):
                events["OmpTaskSwitch"] += 1
            elif isinstance(event, ParameterInt):
                events["ParameterInt"] += 1
            elif isinstance(event, ParameterString):
                events["ParameterString"] += 1
            elif isinstance(event, ParameterType):
                events["ParameterType"] += 1
            elif isinstance(event, ParameterUnsignedInt):
                events["ParameterUnsignedInt"] += 1
            elif isinstance(event, ProgramBegin):
                events["ProgramBegin"] += 1
            elif isinstance(event, ProgramEnd):
                events["ProgramEnd"] += 1
            elif isinstance(event, RmaAcquireLock):
                events["RmaAcquireLock"] += 1
            elif isinstance(event, RmaAtomic):
                events["RmaAtomic"] += 1
            elif isinstance(event, RmaCollectiveBegin):
                events["RmaCollectiveBegin"] += 1
            elif isinstance(event, RmaCollectiveEnd):
                events["RmaCollectiveEnd"] += 1
            elif isinstance(event, RmaGet):
                events["RmaGet"] += 1
            elif isinstance(event, RmaGroupSync):
                events["RmaGroupSync"] += 1
            elif isinstance(event, RmaOpCompleteBlocking):
                events["RmaOpCompleteBlocking"] += 1
            elif isinstance(event, RmaOpCompleteNonBlocking):
                events["RmaOpCompleteNonBlocking"] += 1
            elif isinstance(event, RmaOpCompleteRemote):
                events["RmaOpCompleteRemote"] += 1
            elif isinstance(event, RmaOpTest):
                events["RmaOpTest"] += 1
            elif isinstance(event, RmaPut):
                events["RmaPut"] += 1
            elif isinstance(event, RmaReleaseLock):
                events["RmaReleaseLock"] += 1
            elif isinstance(event, RmaRequestLock):
                events["RmaRequestLock"] += 1
            elif isinstance(event, RmaSync):
                events["RmaSync"] += 1
            elif isinstance(event, RmaTryLock):
                events["RmaTryLock"] += 1
            elif isinstance(event, RmaWaitChange):
                events["RmaWaitChange"] += 1
            elif isinstance(event, RmaWinCreate):
                events["RmaWinCreate"] += 1
            elif isinstance(event, RmaWinDestroy):
                events["RmaWinDestroy"] += 1
            elif isinstance(event, ThreadAcquireLock):
                events["ThreadAcquireLock"] += 1
            elif isinstance(event, ThreadBegin):
                events["ThreadBegin"] += 1
            elif isinstance(event, ThreadCreate):
                events["ThreadCreate"] += 1
            elif isinstance(event, ThreadEnd):
                events["ThreadEnd"] += 1
            elif isinstance(event, ThreadFork):
                events["ThreadFork"] += 1
            elif isinstance(event, ThreadJoin):
                events["ThreadJoin"] += 1
            elif isinstance(event, ThreadReleaseLock):
                events["ThreadReleaseLock"] += 1
            elif isinstance(event, ThreadTaskComplete):
                events["ThreadTaskComplete"] += 1
            elif isinstance(event, ThreadTaskCreate):
                events["ThreadTaskCreate"] += 1
            elif isinstance(event, ThreadTaskSwitch):
                events["ThreadTaskSwitch"] += 1
            elif isinstance(event, ThreadTeamBegin):
                events["ThreadTeamBegin"] += 1
            elif isinstance(event, ThreadTeamEnd):
                events["ThreadTeamEnd"] += 1
            elif isinstance(event, ThreadWait):
                events["ThreadWait"] += 1
            else:
                events["Unnamed"] += 1
        print(events)


event_counts = {
    "AttributeList": 0,
    "Base": 0,
    "BufferFlush": 0,
    "CallingContextEnter": 0,
    "CallingContextLeave": 0,
    "CallingContextSample": 0,
    "CommCreate": 16,
    "CommDestroy": 16,
    "Enter": 9850860,
    "Integral": 0,
    "IoAcquireLock": 0,
    "IoChangeStatusFlags": 0,
    "IoCreateHandle": 0,
    "IoDeleteFile": 0,
    "IoDestroyHandle": 0,
    "IoDuplicateHandle": 0,
    "IoOperationBegin": 0,
    "IoOperationCancelled": 0,
    "IoOperationComplete": 0,
    "IoOperationIssued": 0,
    "IoOperationTest": 0,
    "IoReleaseLock": 0,
    "IoSeek": 0,
    "IoTryLock": 0,
    "Leave": 9850860,
    "MeasurementOnOff": 0,
    "Metric": 676956,
    "MpiCollectiveBegin": 32,
    "MpiCollectiveEnd": 32,
    "MpiIrecv": 32473,
    "MpiIrecvRequest": 152000,
    "MpiIsend": 152000,
    "MpiIsendComplete": 38896,
    "MpiRecv": 70,
    "MpiRequestCancelled": 0,
    "MpiRequestTest": 0,
    "MpiSend": 70,
    "NonBlockingCollectiveComplete": 103,
    "NonBlockingCollectiveRequest": 640,
    "Number": 0,
    "OmpAcquireLock": 0,
    "OmpFork": 0,
    "OmpJoin": 0,
    "OmpReleaseLock": 0,
    "OmpTaskComplete": 0,
    "OmpTaskCreate": 0,
    "OmpTaskSwitch": 0,
    "ParameterInt": 0,
    "ParameterString": 0,
    "ParameterType": 0,
    "ParameterUnsignedInt": 4662342,
    "ProgramBegin": 8,
    "ProgramEnd": 8,
    "RmaAcquireLock": 0,
    "RmaAtomic": 0,
    "RmaCollectiveBegin": 0,
    "RmaCollectiveEnd": 0,
    "RmaGet": 0,
    "RmaGroupSync": 0,
    "RmaOpCompleteBlocking": 124160,
    "RmaOpCompleteNonBlocking": 0,
    "RmaOpCompleteRemote": 0,
    "RmaOpTest": 0,
    "RmaPut": 124160,
    "RmaReleaseLock": 0,
    "RmaRequestLock": 0,
    "RmaSync": 0,
    "RmaTryLock": 0,
    "RmaWaitChange": 0,
    "RmaWinCreate": 0,
    "RmaWinDestroy": 0,
    "ThreadAcquireLock": 0,
    "ThreadBegin": 0,
    "ThreadCreate": 0,
    "ThreadEnd": 0,
    "ThreadFork": 0,
    "ThreadJoin": 0,
    "ThreadReleaseLock": 0,
    "ThreadTaskComplete": 0,
    "ThreadTaskCreate": 0,
    "ThreadTaskSwitch": 0,
    "ThreadTeamBegin": 0,
    "ThreadTeamEnd": 0,
    "ThreadWait": 0,
    "Unnamed": 0,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} path/to/trace.otf2")
        exit(1)

    path = sys.argv[1]
    read_otf2(path)


if __name__ == "__main__":
    main()
