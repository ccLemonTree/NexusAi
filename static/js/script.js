document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('search-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultsContainer = document.getElementById('results');
    const loading = document.getElementById('loading');
    const resultStats = document.querySelector('.result-stats');
    const resultCount = document.getElementById('result-count');
    const searchTime = document.getElementById('search-time');
    const detailModal = document.getElementById('detail-modal');
    const modalBody = document.getElementById('modal-body');
    const closeBtn = document.querySelector('.close-btn');

    // 格式化时间为datetime-local格式（YYYY-MM-DDThh:mm）
    const formatDateTime = (date) => {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        return `${year}-${month}-${day}T${hours}:${minutes}`;
    };

    // 设置默认时间范围（最近24小时）
    const setDefaultTimeRange = () => {
        const end = new Date();
        const start = new Date(end.getTime() - 24 * 60 * 60 * 1000);
        document.getElementById('start_time').value = formatDateTime(start);
        document.getElementById('end_time').value = formatDateTime(end);
    };

    setDefaultTimeRange();

    // 构建查询参数
    const getSearchParams = () => {
        const deviceIds = document.getElementById('device_ids').value
            ? document.getElementById('device_ids').value.split(',').map(id => id.trim())
            : null;

        return {
            text: document.getElementById('text').value || "",
            limit: parseInt(document.getElementById('limit').value)
        };
    };

    // 清空查询条件
    clearBtn.addEventListener('click', () => {
        document.getElementById('text').value = "";
        document.getElementById('pic_path').value = "";
        document.getElementById('device_ids').value = "";
        setDefaultTimeRange();
        document.getElementById('limit').value = "200";
        resultsContainer.innerHTML = "";
        resultStats.style.display = "none";
    });

    // 执行查询
    searchBtn.addEventListener('click', async () => {
        const params = getSearchParams();

        // 验证参数（至少需要文本或图片路径）
        if (!params.text && !params.pic_path) {
            alert("请输入文本描述或图片路径");
            return;
        }

        // 显示加载状态
        loading.style.display = "block";
        resultsContainer.innerHTML = "";
        resultStats.style.display = "none";

        try {
            const startTime = performance.now();
            // 调用API
            const response = await fetch('/ai/seefor/api/search/vlm2vector', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });

            const data = await response.json();
            const endTime = performance.now();
            const duration = ((endTime - startTime) / 1000).toFixed(2);

            // 隐藏加载状态
            loading.style.display = "none";

            if (data.message === "查询成功") {
                // 显示结果统计
                resultCount.textContent = data.data.length;
                searchTime.textContent = duration;
                resultStats.style.display = "block";

                // 渲染结果
                renderResults(data.data);
            } else {
                alert(`查询失败: ${data.message}`);
            }
        } catch (error) {
            loading.style.display = "none";
            console.error("查询出错:", error);
            alert("查询过程中发生错误，请重试");
        }
    });

    // 渲染查询结果
    const renderResults = (results) => {
        if (results.length === 0) {
            resultsContainer.innerHTML = '<p class="no-results">没有找到匹配的结果</p>';
            return;
        }

        resultsContainer.innerHTML = "";
        results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <img src="${item.image_url}" alt="图片" class="result-image">
                <div class="result-info">
                    <p><strong>设备:</strong> ${item.device_name} (${item.device_id})</p>
                    <p><strong>时间:</strong> ${item.timestamp}</p>
                    <p><strong>置信度:</strong> <span class="confidence">${item.confidence}</span></p>
                    <p><strong>通道:</strong> ${item.channel_name}</p>
                </div>
            `;

            // 点击卡片显示详情
            card.addEventListener('click', () => showDetail(item));
            resultsContainer.appendChild(card);
        });
    };

    // 显示详情
    const showDetail = (item) => {
        modalBody.innerHTML = `
            <div class="detail-header">
                <div class="detail-image">
                    <img src="${item.large_image_url}" alt="大图" onerror="this.src='https://via.placeholder.com/400x300?text=图片加载失败'">
                </div>
                <div class="detail-meta">
                    <h3>设备信息</h3>
                    <p><strong>设备ID:</strong> ${item.device_id}</p>
                    <p><strong>设备名称:</strong> ${item.device_name}</p>
                    <p><strong>通道ID:</strong> ${item.channel_id}</p>
                    <p><strong>通道名称:</strong> ${item.channel_name}</p>
                    <p><strong>通道编号:</strong> ${item.channel_number}</p>
                    <p><strong>捕获时间:</strong> ${item.timestamp}</p>
                    <p><strong>置信度:</strong> ${item.confidence}</p>
                </div>
            </div>
            <h3>坐标信息</h3>
            <table class="detail-table">
                <tr>
                    <th>x1</th>
                    <td>${item.x1}</td>
                </tr>
                <tr>
                    <th>x2</th>
                    <td>${item.x2}</td>
                </tr>
                <tr>
                    <th>y1</th>
                    <td>${item.y1}</td>
                </tr>
                <tr>
                    <th>y2</th>
                    <td>${item.y2}</td>
                </tr>
                <tr>
                    <th>记录ID</th>
                    <td>${item.id}</td>
                </tr>
            </table>
        `;
        detailModal.style.display = "block";
    };

    // 关闭详情模态框
    closeBtn.addEventListener('click', () => {
        detailModal.style.display = "none";
    });

    // 点击模态框外部关闭
    window.addEventListener('click', (e) => {
        if (e.target === detailModal) {
            detailModal.style.display = "none";
        }
    });
});