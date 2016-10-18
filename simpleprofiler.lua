--[[

  A simple profiler that measures the time for each events.

--]]

require("torch")

require("utils")

torch.class("SimpleProfiler")

function SimpleProfiler:__init()
  self.clocks = {}
  self.times = {}
end


function SimpleProfiler:reset(event)
  if event ~= nil then
    self.clocks[event] = nil
    self.times[event] = nil
  else
    self.clocks = {}
    self.times = {}
  end
end


function SimpleProfiler:start(event)
  self.clocks[event] = os.clock()
  if self.times[event] == nil then
    self.times[event] = 0
  end
end


function SimpleProfiler:pause(event)
  if self.times[event] ~= nil then
    self.times[event] = self.times[event] + os.clock() - self.clocks[event]
    self.clocks[event] = 0
  end
end


function SimpleProfiler:get_time(event)
  if self.times[event] ~= nil then return self.times[event] else return 0 end
end


function SimpleProfiler:printAll()
  printerr("------------ profiler -------------")
  for k, v in pairs(self.times) do
    printerr("Event " .. k .. " cpu time " .. v)
  end
end

