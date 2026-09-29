import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Model } from './types'

export default function BenchmarkChart({ models }: { models: Model[] }) {
  return <ResponsiveContainer width="100%" height={300}>
    <BarChart data={models} layout="vertical" margin={{ top: 11, left: 25, right: 57, bottom: 4 }} barCategoryGap="43%">
      <CartesianGrid stroke="#3c3d3f" horizontal={false} strokeDasharray="3 5"/>
      <XAxis type="number" domain={[0, 1]} ticks={[0, .25, .5, .75, 1]} tickFormatter={value => `${Math.round(value * 100)}%`} tickLine={false} axisLine={false} tick={{ fill: '#a29b93', fontSize: 10, fontFamily: 'Segoe UI, sans-serif' }}/>
      <YAxis type="category" dataKey="id" width={117} tickLine={false} axisLine={false} tick={{ fill: '#c9c1b8', fontSize: 11, fontWeight: 600, fontFamily: 'Segoe UI, sans-serif' }} tickFormatter={id => id === 'ministral3_3b' ? 'Ministral 3B' : id === 'gemini-3.6-flash' ? 'Gemini Flash' : id.replaceAll('_', ' ')}/>
      <Tooltip cursor={{ fill: '#30302f' }} contentStyle={{ background: '#303032', border: '1px solid #514d49', borderRadius: 8, color: '#e9e3db', fontSize: 11, boxShadow: '0 9px 24px #0b0b0c66' }} formatter={value => `${(Number(value) * 100).toFixed(1)}%`}/>
      <Bar dataKey="accuracy" radius={[0, 4, 4, 0]} barSize={18}>
        {models.map(model => <Cell key={model.id} fill={model.selected_for_v2 ? '#bdad9a' : model.source === 'cloud' ? '#918c85' : '#74736f'}/>)}
        <LabelList dataKey="accuracy" position="right" formatter={(value: unknown) => `${(Number(value) * 100).toFixed(1)}%`} style={{ fill: '#d3cbc2', fontSize: 10, fontWeight: 700, fontFamily: 'Segoe UI, sans-serif' }}/>
      </Bar>
    </BarChart>
  </ResponsiveContainer>
}
